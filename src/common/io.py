"""Shared results writer (Contract 4).

Writes results/<run_name>/test_predictions.csv and results/<run_name>/metrics.json.
Run it from the repository root so that results/ ends up in the right place.
"""
import csv
import json
import os

from .metrics import accuracy, cer, wer

REQUIRED = {"model", "train_subset", "decoding", "params", "train_time_min", "ms_per_image"}


def save_results(run_name, image_paths, refs, hyps, extra):
    """Save predictions and metrics for one run and return the metrics dict.

    extra must contain: model, train_subset, decoding, params, train_time_min, ms_per_image.
    run_name convention: <model>_<subset>_<decoding>, e.g. crnn_p25_beam.
    """
    missing = REQUIRED - set(extra)
    if missing:
        raise ValueError(f"save_results: missing keys in extra: {sorted(missing)}")
    if not (len(image_paths) == len(refs) == len(hyps)):
        raise ValueError("save_results: image_paths, refs and hyps must have the same length")

    out = os.path.join("results", run_name)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "test_predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "ground_truth", "prediction"])
        w.writerows(zip(image_paths, refs, hyps))

    metrics = {
        "run_name": run_name,
        "num_samples": len(refs),
        "cer": cer(refs, hyps),
        "wer": wer(refs, hyps),
        "accuracy": accuracy(refs, hyps),
    }
    metrics.update(extra)
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics
