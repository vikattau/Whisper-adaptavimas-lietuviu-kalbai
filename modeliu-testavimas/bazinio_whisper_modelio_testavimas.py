import whisper
import json
import csv
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip

model = whisper.load_model("large-v3-turbo")

audio = whisper.load_audio("Testas13.mp3")
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

options = whisper.DecodingOptions()
result = model.transcribe(
    "Testas13.mp3",
    language="lt",
    task="transcribe",
    verbose=True
)

print(result["text"])

for seg in result["segments"]:
    print(f'{seg["start"]:.2f} - {seg["end"]:.2f}: {seg["text"]}')

transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

# Dataset
with open("test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

all_refs = []
all_hyps = []
results = []

print("Starting transcription + evaluation...\n")

for i, item in enumerate(data, 1):
    audio_path = item["audio"]
    reference = item["ref"]

    try:
        result = model.transcribe(
            audio_path,
            language="lt",
            task="transcribe"
        )

        hypothesis = result["text"]

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

# Globalios metrikos
final_wer = wer(all_refs, all_hyps)
final_cer = cer(all_refs, all_hyps)

print("==== FINAL RESULTS ====")
print(f"Global WER: {final_wer:.4f}")
print(f"Global CER: {final_cer:.4f}")

csv_file = "results_bazinio.csv"

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file", "reference", "prediction", "wer", "cer"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to {csv_file}")

import torch
import pandas as pd
import whisper
from datasets import load_dataset, Audio
from jiwer import wer, cer
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation, Strip


transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

dataset = "meldynamics/liepa2"

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
        result = model.transcribe(
          audio["array"],
          language="lt"
        )
        hypothesis = result["text"]

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

final_wer = wer(all_refs, all_hyps)
final_cer = cer(all_refs, all_hyps)

print("==== FINAL RESULTS ====")
print(f"Global WER: {final_wer:.4f}")
print(f"Global CER: {final_cer:.4f}")

df = pd.DataFrame(results)
df.to_csv("validation_results_base.csv", index=False, encoding="utf-8-sig")

print("\n=== VALIDATION SUMMARY ===")
print(f"Samples: {len(df)}")
print(f"Average WER: {df['wer'].mean():.4f}")
print(f"Best WER: {df['wer'].min():.4f}")
print(f"Worst WER: {df['wer'].max():.4f}")

print("\n=== WORST EXAMPLES ===")
print(df.sort_values("wer", ascending=False).head(10))

print("\n=== BEST EXAMPLES ===")
print(df.sort_values("wer", ascending=True).head(10))

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace('"', "'")

    return text

# Modelio analize su medicininiais duomenimis
import gc
import csv
import torch
import whisper
import torchaudio
import os

from datasets import load_dataset
from jiwer import wer, cer
from jiwer.transforms import (
    Compose,
    ToLowerCase,
    RemovePunctuation,
    Strip
)

MODEL_NAME = "large-v3-turbo"
DATASET_NAME = "VSSA-SDSA/LT_Medical_S_corpus"
SPLIT = "train"
NUM_SAMPLES = 550

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

model = whisper.load_model(MODEL_NAME, device=DEVICE)

transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT,
    streaming=True,
    token="hf_token"
)

dataset = dataset.take(NUM_SAMPLES)

all_refs = []
all_hyps = []

csv_file = "validation_results_turbo_medical.csv"

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(
        f,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    writer.writerow([
        "id",
        "reference",
        "prediction",
        "wer",
        "cer"
    ])

    for i, sample in enumerate(dataset, 1):

        try:
            audio = sample["audio"]

            waveform = torch.tensor(audio["array"]).float()
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0)

            sampling_rate = audio["sampling_rate"]

            if sampling_rate != 16000:

                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=16000
                )

                waveform = resampler(waveform)

            waveform = waveform.numpy()

            reference = sample["sentence"]

            result = model.transcribe(
                waveform,
                language="lt",
                fp16=torch.cuda.is_available(),
                verbose=False,
                condition_on_previous_text=False,
                temperature=0.0
            )

            hypothesis = result["text"]

            ref_norm = transform(reference)
            hyp_norm = transform(hypothesis)

            sample_wer = wer(ref_norm, hyp_norm)
            sample_cer = cer(ref_norm, hyp_norm)

            all_refs.append(ref_norm)
            all_hyps.append(hyp_norm)

            print("=" * 60)
            print(f"Sample {i}")
            print("-" * 60)
            print("REFERENCE : ", reference)
            print("PREDICTION: ", hypothesis)
            print(f"WER: {sample_wer:.4f}")
            print(f"CER: {sample_cer:.4f}")

            writer.writerow([
                i,
                clean_text(reference),
                clean_text(hypothesis),
                round(sample_wer, 4),
                round(sample_cer, 4)
            ])

            f.flush()

            del waveform
            del result

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:

            print(f"\nERROR processing sample {i}")
            print(e)

global_wer = wer(all_refs, all_hyps)
global_cer = cer(all_refs, all_hyps)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Samples evaluated: {len(all_refs)}")
print(f"Global WER: {global_wer:.4f}")
print(f"Global CER: {global_cer:.4f}")

print("\nResults saved to:")
print(csv_file)