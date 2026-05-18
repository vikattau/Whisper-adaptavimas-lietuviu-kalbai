from data_setup import *
import os
import json
import shutil
import optuna
from huggingface_hub import login, HfApi
from peft import PeftModel

STUDY_NAME = "whisper_optimization_v1.6"
STORAGE = "sqlite:///whisper_optuna.db"

CHECKPOINT_PATH = "outputs/checkpoint-3500"
EXPORT_DIR = "outputs/export_checkpoint_3500"
REPO_ID = "domineeka/whisper-large-lt-v1-opt-checkpoint-3500"


def load_best_params():
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
    )
    return study.best_trial.params


def validate_checkpoint_exists(checkpoint_path):
    print("=== TIKRINAMAS CHECKPOINT ===", flush=True)
    print("Santykinis kelias:", checkpoint_path, flush=True)
    print("Pilnas kelias:", os.path.abspath(checkpoint_path), flush=True)

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Nerastas checkpoint katalogas: {checkpoint_path}")

    files = os.listdir(checkpoint_path)
    print("Checkpoint failai:", files, flush=True)

    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [f for f in required if not os.path.exists(os.path.join(checkpoint_path, f))]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint'e trūksta LoRA adapter failų : {missing}"
        )


def load_model_from_checkpoint(best, checkpoint_path):
    model, processor = load_model(best["r"], best["alpha"])
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    return model, processor


def export_model(model, processor, checkpoint_path, export_dir):

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    model.save_pretrained(export_dir)
    processor.save_pretrained(export_dir)

    metadata = {
        "source_checkpoint": checkpoint_path,
        "global_step": 3500,
        "description": "Exported merged model from LoRA checkpoint-3600 without resuming training",
    }

    with open(os.path.join(export_dir, "checkpoint_export_info.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Modelis išsaugotas   : {export_dir}", flush=True)
    print("Export failai:", os.listdir(export_dir), flush=True)


def push_folder_to_hub(export_dir, repo_id):
    print("=== PUSH Į HUGGING FACE ===", flush=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "Nerastas HF_TOKEN environment variable. "
            "Paleisk prie   sbatch: export HF_TOKEN='hf_xxx'"
        )

    login(token=token)

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Modelis įkeltas Hugging Face repo: {repo_id}", flush=True)

def main():
    print("=== START from_checkpoint.py ===", flush=True)
    validate_checkpoint_exists(CHECKPOINT_PATH)
    print("=== KRAUNU GERIAUSIUS OPTUNA PARAMETRUS ===", flush=True)
    best = load_best_params()
    print("BEST PARAMS:", best, flush=True)

    model, processor = load_model_from_checkpoint(best, CHECKPOINT_PATH)
    export_model(model, processor, CHECKPOINT_PATH, EXPORT_DIR)
    push_folder_to_hub(EXPORT_DIR, REPO_ID)
    print("=== DONE ===", flush=True)

if __name__ == "__main__":
    main()

