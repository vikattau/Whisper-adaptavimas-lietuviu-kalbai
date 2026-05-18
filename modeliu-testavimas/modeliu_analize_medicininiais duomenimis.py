def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace('"', "'")

    return text

import torch
import csv
import gc
import numpy as np
import torchaudio

from datasets import load_dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from peft import PeftModel
from jiwer import wer, cer
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation, Strip

base_model_id = "openai/whisper-large-v3-turbo"
lora_model_id = "domineeka/whisper-large-lt-v1-opt-checkpoint-3500"
dataset_name = "VSSA-SDSA/LT_Medical_S_corpus"

NUM_SAMPLES = 550
DEVICE = 0 if torch.cuda.is_available() else -1

print("Device:", DEVICE)


transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])


processor = AutoProcessor.from_pretrained(base_model_id)

base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

model = PeftModel.from_pretrained(base_model, lora_model_id)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=DEVICE,
    chunk_length_s=30,
    return_timestamps=False,
)

dataset = load_dataset(
    dataset_name,
    split="train",
    streaming=True,
    token="hf_token"
)

dataset = dataset.take(NUM_SAMPLES)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

csv_file = "validation_results.csv"

all_refs = []
all_hyps = []


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
            waveform = np.array(audio["array"], dtype=np.float32)

            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)

            reference = sample["sentence"]

            result = asr(waveform)
            hypothesis = result["text"]

            ref_norm = transform(reference)
            hyp_norm = transform(hypothesis)

            sample_wer = wer(ref_norm, hyp_norm)
            sample_cer = cer(ref_norm, hyp_norm)

            all_refs.append(ref_norm)
            all_hyps.append(hyp_norm)

            print("=" * 60)
            print(f"Sample {i}")
            print("REF:", reference)
            print("HYP:", hypothesis)
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

            del waveform, result
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nERROR at sample {i}")
            print(e)

global_wer = wer(all_refs, all_hyps)
global_cer = cer(all_refs, all_hyps)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Samples: {len(all_refs)}")
print(f"Global WER: {global_wer:.4f}")
print(f"Global CER: {global_cer:.4f}")

print("\nSaved to:", csv_file)