from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import pandas as pd

HIST_BINS = np.linspace(0, 30, 51) 
PRINT_EVERY = 10_000  

dataset = load_dataset(
    "meldynamics/liepa2",
    split="train",
    streaming=True, token = "hf_token"
).decode(False)

dataset = dataset.select_columns(["duration_ms", "is_noise"])

hist_counts = np.zeros(len(HIST_BINS) - 1, dtype=int)
noise_counts = {True: 0, False: 0}

start = time.time()

for i, row in enumerate(dataset, start=1):
    duration_ms = row.get("duration_ms")
    is_noise = row.get("is_noise")

    if duration_ms is None or is_noise is None:
        continue

    duration_s = duration_ms / 1000

    idx = np.searchsorted(HIST_BINS, duration_s, side="right") - 1
    if 0 <= idx < len(hist_counts):
        hist_counts[idx] += 1

    noise_counts[bool(is_noise)] += 1

    if i % PRINT_EVERY == 0:
        print(f"Processed {i} rows in {time.time() - start:.1f}s")

with open("histogram.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["bin_left", "bin_right", "count"])

    for j in range(len(hist_counts)):
        writer.writerow([HIST_BINS[j], HIST_BINS[j + 1], hist_counts[j]])

with open("noise_counts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["is_noise", "count"])
    writer.writerow([False, noise_counts[False]])
    writer.writerow([True, noise_counts[True]])