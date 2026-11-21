"""Shared results writer (Contract 4).

Writes results/<run_name>/test_predictions.csv and results/<run_name>/metrics.json.
Run it from the repository root so that results/ ends up in the right place.

Where this file fits: the last step of every evaluation script. All models save their
results in the same format, so scripts/make_results_table.py can compare them.
"""
# csv / json: the two output formats. os: build folder and file paths.
import csv
import json
import os

from .metrics import accuracy, cer, wer

# Keys that every model must give in `extra`, so the results table has no gaps.
REQUIRED = {"model", "train_subset", "decoding", "params", "train_time_min", "ms_per_image"}

# ===== CONCEPTS USED IN THIS FILE =====
# Contract 4: the agreed results format (folder, file names, required keys) that every model follows.
# run_name: names the results folder, by convention <model>_<subset>_<decoding>, for example crnn_full_greedy.
# CSV: a plain table file. test_predictions.csv has one row per test image (path, truth, prediction).
# JSON: a text format for key/value data. metrics.json holds CER, WER, accuracy and the extra values.
# CER, WER, accuracy: the scores from metrics.py, so all models are measured the same way.
# REQUIRED keys: the extra values every model must report, so the final comparison table has no gaps.
# UTF-8 and ensure_ascii=False: the file encoding that can store Devanagari; the JSON keeps the real letters.
# newline="" for csv: stops extra blank lines when writing on Windows.
# Relative import (from .metrics): imports a sibling file of the same package (src.common).

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: The shared results writer, the last step of evaluating any model. All models save results in one format.
# Q: How is it run?
# A: Not directly. An evaluate script (for example src/crnn/evaluate.py) calls save_results from the repo root.
# Q: What does it read and write?
# A: It reads lists of paths, true texts and predictions plus an extra dict. It writes
#    results/<run_name>/test_predictions.csv and results/<run_name>/metrics.json.
# Q: Why run it from the repository root?
# A: The output folder "results" is a relative path, so it is created in the current folder.


def save_results(run_name, image_paths, refs, hyps, extra):
    """Save predictions and metrics for one run and return the metrics dict.

    extra must contain: model, train_subset, decoding, params, train_time_min, ms_per_image.
    run_name convention: <model>_<subset>_<decoding>, e.g. crnn_p25_beam.

    image_paths, refs, hyps: lists of the same length (image path, true text, predicted text).
    extra: dict of extra values to store in metrics.json (more keys than REQUIRED are allowed).
    Raises ValueError if keys are missing or the lists differ in length.
    """
    # Q: What does it do and why does it exist?
    # A: It checks the inputs, writes the per-image predictions CSV, computes CER, WER and accuracy, and writes
    #    metrics.json. It returns the metrics dict. One writer means every model's results look the same.
    # Q: Why check the keys and the lengths before writing?
    # A: Missing keys would leave gaps in the results table. zip() silently cuts the longer list, so unequal
    #    lists would give wrong rows without any error. Checking first stops this.
    # Set difference: required keys that are not in extra.
    missing = REQUIRED - set(extra)
    if missing:
        raise ValueError(f"save_results: missing keys in extra: {sorted(missing)}")
    # zip() would silently cut the longer list, so check the lengths first.
    if not (len(image_paths) == len(refs) == len(hyps)):
        raise ValueError("save_results: image_paths, refs and hyps must have the same length")

    out = os.path.join("results", run_name)
    os.makedirs(out, exist_ok=True)  # exist_ok: running again overwrites instead of failing

    # One row per test image. utf-8 is needed for Devanagari. newline="" stops blank lines on Windows.
    with open(os.path.join(out, "test_predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "ground_truth", "prediction"])
        w.writerows(zip(image_paths, refs, hyps))

    # Scores come from metrics.py, so all models are measured the same way.
    metrics = {
        "run_name": run_name,
        "num_samples": len(refs),
        "cer": cer(refs, hyps),
        "wer": wer(refs, hyps),
        "accuracy": accuracy(refs, hyps),
    }
    metrics.update(extra)  # add model name, params, timing and any other extra values
    # ensure_ascii=False keeps Devanagari readable in the file instead of \uXXXX escapes.
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics
