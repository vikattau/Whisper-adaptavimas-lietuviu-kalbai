!pip install jiwer

from datasets import load_dataset
from itertools import islice

streamed_dataset = load_dataset(
    "meldynamics/liepa2",
    split="test",
    streaming=True
)

age_groups = set()

for sample in islice(streamed_dataset, 100000):
    age = sample.get("age_group")
    if age is not None:
        age_groups.add(age)

print("Rastos age_group reikšmės:")
for ag in sorted(age_groups):
    print(repr(ag))

from datasets import load_dataset
import pandas as pd
import torch
from tqdm import tqdm
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip
import importlib
import sympy

sympy.printing = importlib.import_module("sympy.printing")

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel

base_model_id = "openai/whisper-large-v3-turbo"
lora_model_id = "domineeka/whisper-large-lt-v1-further-age-gender-2"

TARGET_PER_COMBINATION = 100
MAX_SCANNED = 100000

target_age_groups = [
    "0-12",
    "13-17",
    "18-25",
    "26-60",
    "60+",
]

target_genders = ["male", "female"]

device_name = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(base_model_id)

base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(base_model, lora_model_id)
model.to(device_name)
model.eval()

model_dtype = next(model.parameters()).dtype
print("Device:", device_name)
print("Model dtype:", model_dtype)

forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="lt",
    task="transcribe"
)

transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

streamed_dataset = load_dataset(
    "meldynamics/liepa2",
    split="test",
    streaming=True
)

counts = {
    age: {gender: 0 for gender in target_genders}
    for age in target_age_groups
}

results = []

def all_groups_full(counts, target_n):
    return all(
        counts[age][gender] >= target_n
        for age in counts
        for gender in counts[age]
    )

for idx, sample in enumerate(tqdm(streamed_dataset, total=MAX_SCANNED)):
    if idx >= MAX_SCANNED:
        print(f"Pasiektas MAX_SCANNED={MAX_SCANNED}")
        break

    age_group = sample.get("age_group", "")
    gender = sample.get("gender", "")

    if age_group is None or gender is None:
        continue

    age_group = age_group.strip()
    gender = gender.strip().lower()

    if age_group not in counts:
        continue

    if gender not in counts[age_group]:
        continue

    if counts[age_group][gender] >= TARGET_PER_COMBINATION:
        continue

    try:
        audio = sample["audio"]["array"]
        ref_text = sample.get("text", "")

        features = processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = features["input_features"].to(
            device=device_name,
            dtype=model_dtype
        )

        with torch.no_grad():
            pred_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=256,
            )

        pred_text = processor.batch_decode(
            pred_ids,
            skip_special_tokens=True
        )[0].strip()

        ref_norm = transform(ref_text)
        pred_norm = transform(pred_text)

        wer_score = wer(ref_norm, pred_norm)
        cer_score = cer(ref_norm, pred_norm)

        results.append({
            "idx": idx,
            "gender": gender,
            "age_group": age_group,
            "is_noise": sample.get("is_noise", False),
            "wer": wer_score,
            "cer": cer_score,
            "original_text": ref_text,
            "predicted_text": pred_text,
            "normalized_reference": ref_norm,
            "normalized_prediction": pred_norm,
        })

        counts[age_group][gender] += 1

        if all_groups_full(counts, TARGET_PER_COMBINATION):
            print("Surinkta pakankamai is visu (age × gender). Stabdoma.")
            break

    except Exception as e:
        results.append({
            "idx": idx,
            "gender": gender,
            "age_group": age_group,
            "is_noise": sample.get("is_noise", False),
            "wer": None,
            "cer": None,
            "original_text": sample.get("text", ""),
            "predicted_text": "",
            "normalized_reference": "",
            "normalized_prediction": "",
            "error": str(e),
        })
        print(f"[ERROR] sample {idx}: {e}")

df = pd.DataFrame(results)

df_ok = df[df["wer"].notna()].copy()
df_ok = df_ok.sort_values(["wer", "cer"], ascending=[False, False])

df_errors = df[df["wer"].isna()].copy()

overall_summary = pd.DataFrame([{
    "num_samples": len(df_ok),
    "mean_wer": df_ok["wer"].mean() if len(df_ok) > 0 else None,
    "mean_cer": df_ok["cer"].mean() if len(df_ok) > 0 else None,
}])

by_age = (
    df_ok.groupby("age_group")[["wer", "cer"]]
    .mean()
    .sort_values("wer", ascending=False)
)

by_gender = (
    df_ok.groupby("gender")[["wer", "cer"]]
    .mean()
    .sort_values("wer", ascending=False)
)

by_age_gender = (
    df_ok.groupby(["age_group", "gender"])[["wer", "cer"]]
    .mean()
    .sort_values("wer", ascending=False)
)

count_by_age_gender = df_ok.groupby(["age_group", "gender"]).size()

df_ok.to_csv("evaluation_results_balanced_age_gender.csv", index=False, encoding="utf-8-sig")
overall_summary.to_csv("summary_overall.csv", index=False, encoding="utf-8-sig")
by_age.to_csv("summary_by_age_group.csv", encoding="utf-8-sig")
by_gender.to_csv("summary_by_gender.csv", encoding="utf-8-sig")
by_age_gender.to_csv("summary_by_age_gender.csv", encoding="utf-8-sig")
count_by_age_gender.to_csv("count_by_age_gender.csv", encoding="utf-8-sig")

if len(df_errors) > 0:
    df_errors.to_csv("evaluation_errors.csv", index=False, encoding="utf-8")

print("\n=== Kiek surinkta pagal (age_group × gender) ===")
for age in counts:
    for gender in counts[age]:
        print(f"{age} | {gender}: {counts[age][gender]}")

print("\n=== Vidurkiai pagal (age × gender) ===")
print(by_age_gender)

print("\n=== Bendras vidurkis ===")
print(overall_summary.to_string(index=False))
