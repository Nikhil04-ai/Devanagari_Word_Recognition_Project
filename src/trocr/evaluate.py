"""Evaluate a trained TrOCR checkpoint on the TEST split once and save results with the shared writer.

    python -m src.trocr.evaluate --ckpt checkpoints/trocr_full/best.pt

Run from the repository root. Writes results/trocr_full_greedy/{test_predictions.csv, metrics.json}
(use --decoding beam for results/trocr_full_beam/).

Where it fits in the pipeline: this is the last step. train.py chose best.pt using the validation
set; here that model is run on the test set (data never used for choosing anything) to get the
final numbers: CER, WER, accuracy, speed (ms per image), and CER split by seen / unseen words.
The same shared writer (src/common/io.py) is used by the other models, so all results have the same format.

Decoding modes:
    greedy = pick the most likely character at each step (fast).
    beam   = keep the `num_beams` best partial words at each step (slower, can be more accurate).
"""
# Standard library: argparse (command line), csv (read train words), os (paths), time (timing).
import argparse
import csv
import os
import time

import torch

# Shared project code: result writer, CER metric, seed function.
from src.common.io import save_results
from src.common.metrics import cer
from src.common.seed import set_seed

# Local modules: config, dataset pieces, model helpers, tokenizer, and the scoring loop from train.py.
from .config import load_config
from .dataset import TrOCRWordDataset, get_image_processor, make_loader
from .model import build_model, count_params, generate_ids
from .tokenizer import CharTokenizer
from .train import evaluate_loader


def load_checkpoint(path, device="cpu"):
    """Return (model, tokenizer, cfg, ckpt_dict) from a best.pt written by train.py.

    Input: checkpoint path, device. Returns the model in eval mode on `device`, the tokenizer,
    the config namespace, and the raw checkpoint dict.
    Why the config and tokenizer come from the checkpoint: the model must be rebuilt with the same
    vocabulary and settings it was trained with, whatever the YAML files say now."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # Load on the CPU first; the model is moved to the device at the end.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_config(None, ck["config"])
    tok = CharTokenizer.from_dict(ck["tokenizer"])
    model = build_model(cfg, tok, load_pretrained=False)  # same architecture; weights come from the checkpoint
    model.load_state_dict(ck["model"])
    # eval() switches dropout off.
    return model.to(device).eval(), tok, cfg, ck


@torch.no_grad()
def time_per_image(model, ds, device, num_beams, max_length, n=200, warmup=10):
    """Average ms per image at batch size 1 (GPU if available), after warm-up.

    Inputs: model, dataset, device, decoding settings, n = number of images to time, warmup = images
    to run first and not count. Returns: float, milliseconds per image.
    Why batch size 1: it measures the time of one single prediction, like real use.
    Why warm-up: the first calls are slower (GPU start-up, memory set-up), which would distort the average."""
    # Preload the images as tensors (1, 3, H, W) on the device, so that the timing contains only the model, not file reading.
    px = [ds[i]["pixel_values"][None].to(device) for i in range(min(n + warmup, len(ds)))]
    # If there are very few images, keep at least one for the timed part.
    warm = min(warmup, max(len(px) - 1, 0))
    for x in px[:warm]:
        generate_ids(model, x, num_beams=num_beams, max_length=max_length)
    # GPU work is asynchronous; synchronize() waits until it is really finished, so the clock is correct.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x in px[warm:]:
        generate_ids(model, x, num_beams=num_beams, max_length=max_length)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Seconds -> milliseconds, divided by the number of timed images.
    return 1000 * (time.time() - t0) / max(len(px) - warm, 1)


def main():
    """Command-line entry point: load the checkpoint, predict the test set, compute the metrics, save the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/processed/words")
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--decoding", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N test images (quick check)")
    a = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg, ck = load_checkpoint(a.ckpt, device)
    # Greedy search is the same as a beam width of 1.
    beams = a.num_beams if a.decoding == "beam" else 1

    # Test dataset. first_n=a.limit keeps only the first N rows for a quick check (0 = all rows).
    ds = TrOCRWordDataset(os.path.join(a.data_dir, a.csv), tok, get_image_processor(cfg.pretrained),
                          max_length=cfg.max_length, root=a.image_root, first_n=a.limit)
    # Predict every test image. with_loss=False: no true-label forward pass is needed, only the predicted words.
    res = evaluate_loader(model, make_loader(ds, a.batch_size, False, a.num_workers), tok, cfg, device,
                          amp=device.type == "cuda", num_beams=beams, with_loss=False)
    ms = time_per_image(model, ds, device, beams, cfg.max_length)
    if device.type != "cuda":
        print("WARNING: no GPU found, ms_per_image was measured on CPU")

    # every test word also occurs in training (new writers, not new words): report seen / unseen anyway
    # Set of all words in the training CSV, for a fast "is this word seen in training?" test.
    with open(os.path.join(a.data_dir, "train.csv"), encoding="utf-8", newline="") as f:
        train_words = {r["text"] for r in csv.DictReader(f)}
    # Indices of test samples whose true word is in the training set ("seen") and of the others ("unseen").
    seen = [i for i, r in enumerate(res["refs"]) if r in train_words]
    seen_set = set(seen)
    unseen = [i for i in range(len(res["refs"])) if i not in seen_set]
    # pick(idx, key) -> the entries of res[key] at the given indices.
    pick = lambda idx, key: [res[key][i] for i in idx]  # noqa: E731
    # Extra information stored next to the standard metrics. Rounded for readability;
    # None is used when a group is empty (CER of an empty list is not defined).
    extra = {
        "model": "trocr", "train_subset": a.subset, "decoding": a.decoding,
        "params": count_params(model, trainable_only=False),
        "train_time_min": round(float(ck.get("train_time_min", 0.0)), 2), "ms_per_image": round(ms, 3),
        "best_epoch": ck["epoch"], "n_test_seen_words": len(seen), "n_test_unseen_words": len(unseen),
        "cer_seen_words": cer(pick(seen, "refs"), pick(seen, "hyps")) if seen else None,
        "cer_unseen_words": cer(pick(unseen, "refs"), pick(unseen, "hyps")) if unseen else None,
    }
    # The shared writer saves test_predictions.csv and metrics.json (CER, WER, accuracy, ...) under results/<run_name>/ and returns the metrics dict.
    m = save_results(f"trocr_{a.subset}_{a.decoding}", res["paths"], res["refs"], res["hyps"], extra)
    print({k: m[k] for k in ("run_name", "num_samples", "cer", "wer", "accuracy", "params",
                             "train_time_min", "ms_per_image")})


if __name__ == "__main__":
    main()
