import torch
import json
import csv
import librosa

from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel

base_model_id = "openai/whisper-large-v3-turbo"
lora_model_id = "domineeka/whisper-large-lt-v1-further-age-gender-2"

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

transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

with open("test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_refs = []
all_hyps = []
results = []

for i, item in enumerate(data, 1):
    audio_path = item["audio"]
    reference = item["ref"]

    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        hypothesis = transcribe(audio, sr)

        ref_norm = transform(reference)
        hyp_norm = transform(hypothesis)

        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)

        file_wer = wer(ref_norm, hyp_norm)
        file_cer = cer(ref_norm, hyp_norm)

        print(f"File {i}: {audio_path}")
        print(f"Prediction: {hypothesis}")
        print(f"WER: {file_wer:.4f} | CER: {file_cer:.4f}\n")

        results.append({
            "file": audio_path,
            "reference": reference,
            "prediction": hypothesis,
            "wer": round(file_wer, 4),
            "cer": round(file_cer, 4)
        })

    except Exception as e:
        print(f"Error processing {audio_path}: {e}\n")


final_wer = wer(all_refs, all_hyps)
final_cer = cer(all_refs, all_hyps)

print("==== FINAL RESULTS ====")
print(f"Global WER: {final_wer:.4f}")
print(f"Global CER: {final_cer:.4f}")

csv_file = "results_further_liepa2.csv"

with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "reference", "prediction", "wer", "cer"]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {csv_file}")