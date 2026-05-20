!pip install evaluate
!pip install jiwer

!pip uninstall -y torchao

import torch
import json
import csv
import librosa

from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel

base_model_id = "openai/whisper-large-v3-turbo"
lora_model_id = "domineeka/whisper-large-lt-v1-further-age-gender-2"
dataset = "meldynamics/liepa2"

processor = AutoProcessor.from_pretrained(base_model_id)

base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

model = PeftModel.from_pretrained(base_model, lora_model_id)

model.to("cuda" if torch.cuda.is_available() else "cpu")

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

def transcribe(audio, sr):
    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True
    )

    input_features = inputs.input_features.to(model.device)

    input_features = input_features.to(model.dtype)

    predicted_ids = model.generate(
        input_features,
        language="lt"
    )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription

import torch
import pandas as pd
from datasets import load_dataset, Audio
from jiwer import wer, cer
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation, Strip

# ===== NORMALIZATION =====
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

# ===== LOAD DATASET =====
dataset = load_dataset(dataset, split="test", streaming=True)
dataset = dataset.skip(1000).take(1000)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

all_refs = []
all_hyps = []
results = []

for i, sample in enumerate(dataset, 1):
    audio = sample["audio"]
    reference = sample["text"]

    try:
        hypothesis = transcribe(audio["array"], audio["sampling_rate"])

        ref_norm = transform(reference)
        hyp_norm = transform(hypothesis)

        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)

        file_wer = wer(ref_norm, hyp_norm)
        file_cer = cer(ref_norm, hyp_norm)

        print(f"File {i}")
        print(f"Prediction: {hypothesis}")
        print(f"WER: {file_wer:.4f} | CER: {file_cer:.4f}\n")

        results.append({
            "id": i,
            "reference": reference,
            "prediction": hypothesis,
            "wer": round(file_wer, 4),
            "cer": round(file_cer, 4)
        })

    except Exception as e:
        print(f"Error processing sample {i}: {e}")

# ===== FINAL =====
final_wer = wer(all_refs, all_hyps)
final_cer = cer(all_refs, all_hyps)

print("==== FINAL RESULTS ====")
print(f"Global WER: {final_wer:.4f}")
print(f"Global CER: {final_cer:.4f}")

# ===== SAVE =====
df = pd.DataFrame(results)
df.to_csv("validation_results_trecdalio.csv", index=False, encoding="utf-8-sig")

# ===== SUMMARY =====
print("\n=== VALIDATION SUMMARY ===")
print(f"Samples: {len(df)}")
print(f"Average WER: {df['wer'].mean():.4f}")
print(f"Best WER: {df['wer'].min():.4f}")
print(f"Worst WER: {df['wer'].max():.4f}")

# ===== ERROR ANALYSIS =====
print("\n=== WORST EXAMPLES ===")
print(df.sort_values("wer", ascending=False).head(10))

print("\n=== BEST EXAMPLES ===")
print(df.sort_values("wer", ascending=True).head(10))

clean = []
noise = []

clean_results = []
noise_results = []

for sample in dataset:
    audio = sample["audio"]
    ref = sample["text"]

    hyp = transcribe(audio["array"], audio["sampling_rate"])

    ref_norm = transform(ref)
    hyp_norm = transform(hyp)

    row = {
        "reference": ref,
        "prediction": hyp,
        "wer": wer(ref_norm, hyp_norm),
        "cer": cer(ref_norm, hyp_norm)
    }

    if sample["is_noise"]:
        noise_results.append(row)
    else:
        clean_results.append(row)

clean_df = pd.DataFrame(clean_results)

print("\n=== CLEAN VALIDATION SUMMARY ===")
print(f"Samples: {len(clean_df)}")
print(f"Average WER: {clean_df['wer'].mean():.4f}")
print(f"Best WER: {clean_df['wer'].min():.4f}")
print(f"Worst WER: {clean_df['wer'].max():.4f}")

print("\n=== CLEAN WORST EXAMPLES ===")
print(clean_df.sort_values("wer", ascending=False).head(10))

print("\n=== CLEAN BEST EXAMPLES ===")
print(clean_df.sort_values("wer", ascending=True).head(10))


noise_df = pd.DataFrame(noise_results)

clean_df.to_csv("validation_results.csv", index=False, encoding="utf-8-sig")


print("\n=== NOISE VALIDATION SUMMARY ===")
print(f"Samples: {len(noise_df)}")
print(f"Average WER: {noise_df['wer'].mean():.4f}")
print(f"Best WER: {noise_df['wer'].min():.4f}")
print(f"Worst WER: {noise_df['wer'].max():.4f}")


print("\n=== NOISE WORST EXAMPLES ===")
print(noise_df.sort_values("wer", ascending=False).head(10))

print("\n=== NOISE BEST EXAMPLES ===")
print(noise_df.sort_values("wer", ascending=True).head(10))



import torch
import pandas as pd
from datasets import load_dataset, Audio
from jiwer import wer, cer
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation, Strip

# ===== NORMALIZATION =====
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

# ===== LOAD DATASET =====
dataset = load_dataset(dataset, split="test", streaming=True)
dataset = dataset.skip(1000).take(1000)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

all_refs = []
all_hyps = []
results = []

for i, sample in enumerate(dataset, 1):
    audio = sample["audio"]
    reference = sample["text"]

    try:
        hypothesis = transcribe(audio["array"], audio["sampling_rate"])

        ref_norm = transform(reference)
        hyp_norm = transform(hypothesis)

        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)

        file_wer = wer(ref_norm, hyp_norm)
        file_cer = cer(ref_norm, hyp_norm)

        print(f"File {i}")
        print(f"Prediction: {hypothesis}")
        print(f"WER: {file_wer:.4f} | CER: {file_cer:.4f}\n")

        results.append({
            "id": i,
            "reference": reference,
            "prediction": hypothesis,
            "wer": round(file_wer, 4),
            "cer": round(file_cer, 4)
        })

    except Exception as e:
        print(f"Error processing sample {i}: {e}")

# ===== FINAL =====
final_wer = wer(all_refs, all_hyps)
final_cer = cer(all_refs, all_hyps)

print("==== FINAL RESULTS ====")
print(f"Global WER: {final_wer:.4f}")
print(f"Global CER: {final_cer:.4f}")

# ===== SAVE =====
df = pd.DataFrame(results)
df.to_csv("validation_results.csv", index=False, encoding="utf-8-sig")

# ===== SUMMARY =====
print("\n=== VALIDATION SUMMARY ===")
print(f"Samples: {len(df)}")
print(f"Average WER: {df['wer'].mean():.4f}")
print(f"Best WER: {df['wer'].min():.4f}")
print(f"Worst WER: {df['wer'].max():.4f}")

# ===== ERROR ANALYSIS =====
print("\n=== WORST EXAMPLES ===")
print(df.sort_values("wer", ascending=False).head(10))

print("\n=== BEST EXAMPLES ===")
print(df.sort_values("wer", ascending=True).head(10))
