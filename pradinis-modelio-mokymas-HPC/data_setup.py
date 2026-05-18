#ENV FIX
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#Imports
import torch, gc, optuna
from dataclasses import dataclass
from unsloth import FastModel, is_bf16_supported
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

DATASET = "meldynamics/liepa2"
LANGUAGE = "Lithuanian"

wer_metric = evaluate.load("wer")

#Model loader
def load_model(r=8, alpha=16):
    orig_warn_flag = os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1")
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"
    try:
        model, processor = FastModel.from_pretrained(
            model_name="openai/whisper-large-v3-turbo",
            load_in_4bit=True,
            auto_model=WhisperForConditionalGeneration,
            whisper_language=LANGUAGE,
            whisper_task="transcribe",
        )
    finally:
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = orig_warn_flag

    model = FastModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        task_type=None,
    )

    model.config.use_cache = False
    return model, processor


#Preprocessing
MAX_LABEL_LEN = 447

def preprocess_batch(examples, processor):
    audio_col = "ogg" if "ogg" in examples else "audio"
    arrays = [x["array"] for x in examples[audio_col]]

    feats = processor.feature_extractor(
        arrays,
        sampling_rate=16000,
        return_tensors="np"
    ).input_features

    labels = processor.tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LABEL_LEN
    ).input_ids

    return {"input_features": [f for f in feats], "labels": labels}


#Collator
@dataclass
class SpeechCollator:
    processor: object

    def __call__(self, features):
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")

        label_ids = [{"input_ids": f["labels"]} for f in features]
        padded = self.processor.tokenizer.pad(label_ids, return_tensors="pt")

        labels = padded["input_ids"].masked_fill(
            padded["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


#Metric
def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    label_ids = label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


#Dataset helpers
print("Preparing eval dataset once…")

_init_model, _init_processor = load_model(8, 16)

eval_dataset = (
    load_dataset(DATASET, split="test[:1000]")
    .map(
        lambda x: preprocess_batch(x, _init_processor),
        batched=True,
        batch_size=4,
        remove_columns=["audio", "text"],
    )
)
print("Preparing eval dataset done")
del _init_model
torch.cuda.empty_cache()
gc.collect()


def make_train(processor, bs):
    ds = load_dataset(DATASET, streaming=True, split="train")
    ds = ds.shuffle(seed=42, buffer_size=1000)

    return ds.map(
        lambda x: preprocess_batch(x, processor),
        batched=True,
        batch_size=bs,
        remove_columns=["audio", "text"],
    )