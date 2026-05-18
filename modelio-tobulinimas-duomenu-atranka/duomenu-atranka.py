import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import numpy as np
import optuna
import torch
import gc
import pandas as pd
from unsloth import FastModel, is_bf16_supported
from dataclasses import dataclass
from datasets import load_from_disk, concatenate_datasets
from huggingface_hub import login
from peft import PeftModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers import WhisperForConditionalGeneration


from data_setup import (
    compute_metrics,
    eval_dataset,
    MAX_LABEL_LEN,
    LANGUAGE,
)
CACHED_DATASET_DIR = "cache/liepa2_train_raw"

HF_TOKEN = "HF_token"
BASE_MODEL_REPO = "openai/whisper-large-v3-turbo"

LOAD_HF_REPO = "domineeka/whisper-large-lt-v1-opt-checkpoint-3500"
PUSH_HF_REPO = "domineeka/whisper-large-lt-v1-further-age-gender-2"

INITIAL_OUTPUT_DIR = "outputs"
EVAL_CSV_DIR = "."

SAMPLING_DIMENSIONS = ["age_gender"]
AGE_GENDER_CSV = "summary_by_age_gender.csv"
WER_WEIGHT = 0.7
CER_WEIGHT = 0.3

INITIAL_TRAIN_SEED = 42
RANDOM_SEED = 42
OUTPUT_DIR = "outputs_further_age_gender_2"

FURTHER_MAX_STEPS = 20000
LEARNING_RATE_MULTIPLIER = 1.0

RECONSTRUCTION_BS = 2
RECONSTRUCTION_GA = 4


def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        raise RuntimeError(
            f"\n[ERROR] Initial training output directory not found: {output_dir}\n"
            f"  Make sure asr_train.py has been run and saved at least one checkpoint."
        )

    pattern = re.compile(r"^checkpoint-(\d+)$")
    checkpoints = []

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        match = pattern.match(name)
        if match and os.path.isdir(path):
            checkpoints.append((int(match.group(1)), name))

    if not checkpoints:
        raise RuntimeError(
            f"\n[ERROR] No checkpoints found in: {output_dir}\n"
            f"  Expected subdirectories named checkpoint-<step>."
        )

    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_name = checkpoints[-1]
    latest_path = os.path.join(output_dir, latest_name)

    print(f"Found {len(checkpoints)} checkpoint(s) in '{output_dir}':")
    for step, name in checkpoints:
        marker = " <-- latest" if step == latest_step else ""
        print(f"  {name}{marker}")

    return latest_path, latest_step


def get_steps_from_checkpoint(checkpoint_path):
    state_file = os.path.join(checkpoint_path, "trainer_state.json")

    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        step = state.get("global_step")
        if step is not None:
            print(f"Steps completed (from trainer_state.json): {step}")
            return int(step)

    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    if match:
        step = int(match.group(1))
        print(f"Steps completed (from folder name): {step}")
        return step

    raise RuntimeError(
        f"\n[ERROR] Could not determine step count from checkpoint: {checkpoint_path}"
    )


def get_seen_indices(total_samples, steps_completed, batch_size, gradient_accumulation_steps, seed):
    samples_seen = steps_completed * batch_size * gradient_accumulation_steps

    print("\nReconstructing seen indices:")
    print(f"  steps_completed : {steps_completed}")
    print(f"  batch_size      : {batch_size}")
    print(f"  grad_accum      : {gradient_accumulation_steps}")
    print(f"  samples_seen    : {samples_seen}")
    print(f"  total_samples   : {total_samples}")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(total_samples)

    if samples_seen >= total_samples:
        print("WARNING: samples_seen >= total_samples. Using full dataset as already seen.")
        return set(range(total_samples))

    seen_set = set(shuffled_indices[:samples_seen].tolist())
    print(f"  Unique seen indices: {len(seen_set)}")
    return seen_set


def exclude_seen_samples(dataset, seen_indices):
    total = len(dataset)
    unseen_indices = [i for i in range(total) if i not in seen_indices]
    unseen_ds = dataset.select(unseen_indices)

    print(f"Excluded {total - len(unseen_ds)} seen rows, {len(unseen_ds)} unseen rows remain.")
    return unseen_ds


def load_wer_from_csv(csv_path, group_col):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Required CSV not found: {csv_path}\n"
            f"  Run the evaluation notebook first to generate WER summary CSVs, then re-run this script."
        )

    df = pd.read_csv(csv_path)

    if group_col not in df.columns:
        df = df.reset_index()

    if group_col not in df.columns:
        raise ValueError(
            f"\n[ERROR] Column '{group_col}' not found in {csv_path}\n"
            f"  Available columns: {list(df.columns)}"
        )

    if "wer" not in df.columns:
        raise ValueError(
            f"\n[ERROR] Column 'wer' not found in {csv_path}\n"
            f"  Available columns: {list(df.columns)}"
        )

    wer_dict = dict(zip(df[group_col], df["wer"]))
    print(f"Loaded WER from {csv_path}: {wer_dict}")
    return wer_dict

def load_age_gender_score_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n[ERROR] Required CSV not found: {csv_path}\n"
            f"Expected columns: age_group, gender, wer, cer"
        )

    df = pd.read_csv(csv_path)

    required = {"age_group", "gender", "wer", "cer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"\n[ERROR] Missing columns in {csv_path}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    wer_max = df["wer"].max()
    cer_max = df["cer"].max()

    df["wer_norm"] = df["wer"] / wer_max if wer_max > 0 else 1.0
    df["cer_norm"] = df["cer"] / cer_max if cer_max > 0 else 1.0

    df["score"] = WER_WEIGHT * df["wer_norm"] + CER_WEIGHT * df["cer_norm"]

    score_dict = {
        f"{row['age_group']},{row['gender']}": round(row["score"], 4)
        for _, row in df.iterrows()
    }

    print(f"Loaded age+gender WER/CER scores from {csv_path}:")
    print(score_dict)

    return score_dict

def compute_weights(wer_dict):
    if not wer_dict:
        return {}

    wer_max = max(wer_dict.values())
    if wer_max == 0:
        return {k: 1.0 for k in wer_dict}

    return {k: round(v / wer_max, 4) for k, v in wer_dict.items()}


def sample_by_group(dataset, group_col, weights, seed):
    subsets = []
    unique_vals = set(dataset.unique(group_col))

    for val in unique_vals:
        subset = dataset.filter(lambda x, v=val: x[group_col] == v)
        frac = weights.get(val, 1.0)
        total = len(subset)

        if total == 0:
            continue

        if frac >= 1.0:
            kept_subset = subset
            kept = total
        else:
            n_keep = max(1, int(total * frac))
            kept_subset = subset.shuffle(seed=seed).select(range(n_keep))
            kept = n_keep

        subsets.append(kept_subset)
        print(f"  [{group_col}={val!r}] weight={frac:.4f} kept {kept}/{total} rows")

    if not subsets:
        raise RuntimeError(f"No subsets created for group_col='{group_col}'.")

    return concatenate_datasets(subsets).shuffle(seed=seed)


def build_further_train_dataset(steps_completed, batch_size, gradient_accumulation_steps, dimensions, seed=RANDOM_SEED):
    print(f"\nLoading cached RAW dataset from: {CACHED_DATASET_DIR}")
    ds = load_from_disk(CACHED_DATASET_DIR)
    print(f"Total rows in cache: {len(ds)}")
    print(f"Columns in cache: {ds.column_names}")

    print("\nExcluding samples seen during initial training...")
    seen_indices = get_seen_indices(
        total_samples=len(ds),
        steps_completed=steps_completed,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=INITIAL_TRAIN_SEED,
    )
    ds = exclude_seen_samples(ds, seen_indices)

    age_gender_weights = {}

    if "age_gender" in dimensions:
        ds = ds.filter(
            lambda x: x["age_group"] and x["age_group"] != "" and x["gender"] and x["gender"] != ""
        )
        print(f"After removing empty age_group/gender: {len(ds)} rows")

        ds = ds.map(lambda x: {
            "age_gender": f"{x['age_group']},{x['gender']}"
        })

        age_gender_scores = load_age_gender_score_from_csv(
            os.path.join(EVAL_CSV_DIR, AGE_GENDER_CSV)
        )

        age_gender_weights = compute_weights(age_gender_scores)

    print("\nComputed sampling weights:")
    print("  Age+Gender: ", age_gender_weights)

    dim_map = {
        "age_gender": ("age_gender", age_gender_weights),
    }

    for dim in dimensions:
        if dim not in dim_map:
            raise ValueError(f"Unknown dimension: {dim!r}. Choose from {list(dim_map.keys())}")

        col, weights = dim_map[dim]

        if col not in ds.column_names:
            print(f"[SKIP] Column '{col}' not in dataset -- skipping '{dim}' dimension")
            continue

        print(f"\nApplying sampling for '{dim}' (column='{col}')...")
        ds = sample_by_group(ds, col, weights, seed)

    print(f"\nFinal further-training RAW dataset size: {len(ds)} rows")
    print(f"Final RAW columns: {ds.column_names}")

    return ds, {
        "age_gender": age_gender_weights,
    }


def load_trainable_lora_adapter_from_hub():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("[WARN] HF_TOKEN env variable not set. Loading/pushing will work only for public repos or cached auth.")

    orig_warn = os.environ.get("UNSLOTH_WARN_UNINITIALIZED", "1")
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = "0"

    try:
        base_model, processor = FastModel.from_pretrained(
            model_name=BASE_MODEL_REPO,
            load_in_4bit=True,
            auto_model=WhisperForConditionalGeneration,
            whisper_language=LANGUAGE,
            whisper_task="transcribe",
        )
    finally:
        os.environ["UNSLOTH_WARN_UNINITIALIZED"] = orig_warn

    try:
        model = PeftModel.from_pretrained(
            base_model,
            LOAD_HF_REPO,
            is_trainable=True,
        )
    except Exception as e:
        raise RuntimeError(
            "\n[ERROR] Nepavyko užkrauti LoRA adapterio iš HF repo.\n"
            f"  LOAD_HF_REPO = {LOAD_HF_REPO}\n"
            f"\nOriginal error: {repr(e)}"
        ) from e

    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found.")
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    return model, processor

def objective_further(trial, train_dataset, eval_dataset, processor):
    lr    = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    bs    = trial.suggest_categorical("bs", [2, 4])
    ga    = trial.suggest_categorical("ga", [2, 4, 8])

    model = None
    trainer = None
    try:

        model, _ = load_trainable_lora_adapter_from_hub()

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"optuna_trial_{trial.number}",
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            max_steps=30,
            learning_rate=lr,
            eval_strategy="steps",
            eval_steps=10,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            bf16=is_bf16_supported(),
            fp16=not is_bf16_supported(),
            predict_with_generate=True,
            generation_max_length=MAX_LABEL_LEN,
            remove_unused_columns=False,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=HybridSpeechCollator(processor),
            compute_metrics=lambda p: compute_metrics(
                p,
                processor,
            ),
            args=training_args,
        )

        trainer.train()

        metrics = trainer.evaluate()

        return metrics["eval_wer"]

    except Exception as e:

        print(f"[ERROR] Trial failed: {repr(e)}")

        return 1.0

    finally:

        del model
        del trainer

        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class HybridSpeechCollator:
    processor: object

    def __call__(self, features):
        if not features:
            return {}

        if "audio" in features[0] and "text" in features[0]:
            arrays = [f["audio"]["array"] for f in features]

            input_features = self.processor.feature_extractor(
                arrays,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features

            tokenized = self.processor.tokenizer(
                [f["text"] for f in features],
                truncation=True,
                max_length=MAX_LABEL_LEN,
                padding=True,
                return_tensors="pt",
            )

            labels = tokenized["input_ids"].masked_fill(
                tokenized["attention_mask"].ne(1), -100
            )

            return {
                "input_features": input_features,
                "labels": labels,
            }

        if "input_features" in features[0] and "labels" in features[0]:
            input_feats = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")

            label_ids = [{"input_ids": f["labels"]} for f in features]
            padded = self.processor.tokenizer.pad(label_ids, return_tensors="pt")

            labels = padded["input_ids"].masked_fill(
                padded["attention_mask"].ne(1), -100
            )
            batch["labels"] = labels
            return batch

        raise ValueError(f"Unexpected feature keys in batch: {list(features[0].keys())}")


class FurtherTrainingCallback(TrainerCallback):
    def __init__(self, save_every=200):
        self.save_every = save_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            print(f"[Callback] Checkpoint at step {state.global_step}: {ckpt}")

def main():

    print(f"\nScanning for latest local checkpoint in: {INITIAL_OUTPUT_DIR}")
    checkpoint_path, _ = find_latest_checkpoint(INITIAL_OUTPUT_DIR)

    steps_completed = get_steps_from_checkpoint(checkpoint_path)

    print(
        f"Initial training checkpoint: {checkpoint_path} "
        f"(step {steps_completed})"
    )

    train_dataset, all_weights = build_further_train_dataset(
        steps_completed=steps_completed,
        batch_size=RECONSTRUCTION_BS,
        gradient_accumulation_steps=RECONSTRUCTION_GA,
        dimensions=SAMPLING_DIMENSIONS,
        seed=RANDOM_SEED,
    )

    print(f"\nFinal dataset size: {len(train_dataset)}")

    _, processor = load_trainable_lora_adapter_from_hub()
    torch.cuda.empty_cache()
    gc.collect()

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///whisper_optuna.db",
        study_name="whisper_optimization_optuna_v2",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective_further(
            trial,
            train_dataset,
            eval_dataset,
            processor, 
        ),
        n_trials=10,
    )

    best = study.best_trial.params

    print("\nBest hyperparams:")
    print(best)

    print(f"\nLoading base model: {BASE_MODEL_REPO}")
    model, processor = load_trainable_lora_adapter_from_hub()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    learning_rate = float(best["lr"]) * LEARNING_RATE_MULTIPLIER

    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=HybridSpeechCollator(processor),
        compute_metrics=lambda p: compute_metrics(p, processor),
        callbacks=[FurtherTrainingCallback(save_every=200)],
        args=Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=best["bs"],
            gradient_accumulation_steps=best["ga"],
            max_steps=FURTHER_MAX_STEPS,
            learning_rate=learning_rate,
            eval_strategy="steps",
            eval_steps=200,
            logging_steps=200,
            report_to="none",
            save_strategy="steps",
            save_steps=200,
            save_total_limit=5,
            predict_with_generate=True,
            generation_max_length=MAX_LABEL_LEN,
            bf16=is_bf16_supported(),
            fp16=not is_bf16_supported(),
            remove_unused_columns=False,
        ),
    )

    config = {
        "base_model": BASE_MODEL_REPO,
        "source_lora_adapter": LOAD_HF_REPO,
        "local_checkpoint_for_seen_sample_estimation": checkpoint_path,
        "steps_completed_for_seen_sample_estimation": steps_completed,
        "sampling_dimensions": SAMPLING_DIMENSIONS,
        "all_weights": all_weights,
        "age_gender_csv": AGE_GENDER_CSV,
        "wer_weight": WER_WEIGHT,
        "cer_weight": CER_WEIGHT,
        "eval_csv_dir": EVAL_CSV_DIR,
        "best_hyperparams": best,
        "learning_rate_used": learning_rate,
        "learning_rate_multiplier": LEARNING_RATE_MULTIPLIER,
        "further_max_steps": FURTHER_MAX_STEPS,
        "train_rows_raw_after_sampling": len(train_dataset),
        "cached_dataset_dir": CACHED_DATASET_DIR,
        "preprocessing_mode": "on_the_fly_in_collator",
        "push_model_repo": PUSH_HF_REPO,
    }

    cfg_path = os.path.join(OUTPUT_DIR, "further_train_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {cfg_path}")

    print("\nStarting further training...")
    trainer.train()

    final_model_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    print(f"Final trainable adapter saved locally to: {final_model_dir}")

    login(token=HF_TOKEN)
    model.push_to_hub(PUSH_HF_REPO)
    processor.push_to_hub(PUSH_HF_REPO)
    #trainer.push_to_hub(PUSH_HF_REPO)
    print(f"Final adapter pushed to HuggingFace: {PUSH_HF_REPO}")


if __name__ == "__main__":
    main()
