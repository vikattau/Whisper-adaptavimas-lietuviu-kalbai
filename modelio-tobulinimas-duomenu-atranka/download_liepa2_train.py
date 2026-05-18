import os
import gc
import json
from datasets import load_dataset

DATASET = "meldynamics/liepa2"
OUTPUT_DIR = "cache/liepa2_train_raw"
KEEP_COLUMNS = ["audio", "text", "gender", "age_group", "is_noise"]


def main():
    print("=== LIEPA2 train parsisiuntimas pradėtas ===", flush=True)
    print(f"Dataset: {DATASET}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)

    ds = load_dataset(DATASET, split="train")
    print(f"Train įrašų skaičius: {len(ds)}", flush=True)
    print(f"Originalūs stulpeliai: {ds.column_names}", flush=True)

    keep_cols = [c for c in KEEP_COLUMNS if c in ds.column_names]
    print(f"Bus palikti stulpeliai: {keep_cols}", flush=True)

    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
        print(f"Pašalinti stulpeliai: {remove_cols}", flush=True)

    print(f"Liko stulpeliai: {ds.column_names}", flush=True)

    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    print("Pradedamas save_to_disk...", flush=True)
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Išsaugota į: {OUTPUT_DIR}", flush=True)

    info = {
        "dataset": DATASET,
        "rows": len(ds),
        "columns": ds.column_names,
        "output_dir": OUTPUT_DIR,
    }

    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("dataset_info.json išsaugotas", flush=True)
    print("Pavyzdinio įrašo raktai:", ds[0].keys(), flush=True)

    del ds
    gc.collect()

    print("=== Baigta ===", flush=True)


if __name__ == "__main__":
    main()