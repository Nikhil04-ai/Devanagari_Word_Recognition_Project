"""Evaluate a trained CRNN on the TEST split once and save results with the shared writer.

    python -m src.crnn.evaluate --ckpt checkpoints/crnn_full/best.pt

Run from the repository root. Writes results/crnn_full_greedy/{test_predictions.csv, metrics.json}.

Where this file fits in the pipeline:
    train.py (best.pt) -> evaluate.py (this file) -> results/<run>/ -> scripts/make_results_table.py.
The test set is used only here, once, after all choices (epochs, best checkpoint) were made on the
validation set. This keeps the test numbers honest.

Besides CER / WER / accuracy it reports the model size, the training time, the speed per image,
and the CER split into test words that were seen in training and words that were not.
"""
# Standard library: command-line flags, CSV reading, file paths, timing.
import argparse
import csv
import os
import time

# torch for the model, PIL to open images, nn for the CTC loss object.
import torch
from PIL import Image
from torch import nn

# Shared results writer and metric.
from src.common.io import save_results
from src.common.metrics import cer

# Code of this package. evaluate_loader is reused from train.py so validation and test run the same way.
from .dataset import CharVocab, WordDataset, collate_fn, load_image
from .model import CRNN, count_params
from .train import evaluate_loader


def load_checkpoint(path, device):
    """Load a checkpoint written by train.py.

    path: .pt file. device: torch device to place the model on.
    Returns (model in eval mode, CharVocab, the raw checkpoint dict).
    """
    # weights_only=False: the file holds plain Python objects (cfg, chars) as well as tensors.
    # Only load checkpoints that you trained yourself.
    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = CharVocab(ck["chars"])  # the exact character list used in training, so indices match
    cfg = ck["cfg"]
    # Build the same architecture as in training (the layer sizes must match the saved weights).
    model = CRNN(len(vocab), hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()  # dropout off, BatchNorm uses running statistics
    return model, vocab, ck


@torch.no_grad()
def ms_per_image(model, paths, root, device, invert, n=200, warmup=10):
    """Average time to run the network on one image (batch size 1).

    paths: image paths (relative to root). n: how many images to time. warmup: extra images run first
    and not timed. Returns milliseconds per image (float).
    Only the network forward pass is timed. Image loading and decoding are not included.
    """
    # Load all images first, so disk reading is not part of the measured time.
    imgs = [load_image(Image.open(os.path.join(root, p)), invert=invert).unsqueeze(0).to(device)
            for p in paths[: n + warmup]]  # each tensor is (1, 1, 64, W)
    # Warm-up runs: the first calls are slower (GPU start-up, kernel setup), so they are not measured.
    for x in imgs[:warmup]:
        model(x)
    # GPU work runs in the background. synchronize() waits for it, so the clock is correct.
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x in imgs[warmup:]:
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / max(len(imgs) - warmup, 1)  # seconds -> ms, per image


def main():
    """Command-line entry point: load the checkpoint, run the test set, save results, print a summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)  # checkpoint to test, normally best.pt
    ap.add_argument("--data_dir", default="data/processed/words")
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--image_root", default=".")
    # subset is only a label for the run name (results/crnn_<subset>_greedy), it does not select data.
    ap.add_argument("--subset", default="full")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N test images (quick check)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, ck = load_checkpoint(a.ckpt, device)
    # Use the same colour setting as in training (older checkpoints may not have the key).
    invert = ck["cfg"].get("invert", False)

    ds = WordDataset(os.path.join(a.data_dir, a.csv), vocab, root=a.image_root, invert=invert)
    if a.limit:
        ds.samples = ds.samples[: a.limit]
    # shuffle=False keeps the test order fixed, so results are easy to compare.
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.num_workers, collate_fn=collate_fn)
    # blank=0 is the same as CharVocab.BLANK. The loss value is not reported here, only CER / WER.
    res = evaluate_loader(model, loader, vocab, device, nn.CTCLoss(blank=0, zero_infinity=True))

    # the dataset has few distinct words: also report test words seen / unseen in training
    # Set of all words that appear in the training labels.
    with open(os.path.join(a.data_dir, "train.csv"), encoding="utf-8", newline="") as f:
        train_words = {r["text"] for r in csv.DictReader(f)}
    # Indices of test samples whose true word was / was not in training.
    seen = [i for i, r in enumerate(res["refs"]) if r in train_words]
    unseen = [i for i in range(len(res["refs"])) if i not in set(seen)]
    # Helper: take the entries of res[key] at the given indices.
    pick = lambda idx, key: [res[key][i] for i in idx]  # noqa: E731
    # Values saved next to the standard metrics. The first six keys are required by save_results.
    extra = {
        "model": "crnn", "train_subset": a.subset, "decoding": "greedy",
        "params": count_params(model), "train_time_min": round(ck.get("train_time_min", 0.0), 2),
        "ms_per_image": round(ms_per_image(model, res["paths"], a.image_root, device, invert), 3),
        "best_epoch": ck["epoch"], "n_test_seen_words": len(seen), "n_test_unseen_words": len(unseen),
        # None (null in JSON) when a group is empty, because CER of nothing is undefined.
        "cer_seen_words": cer(pick(seen, "refs"), pick(seen, "hyps")) if seen else None,
        "cer_unseen_words": cer(pick(unseen, "refs"), pick(unseen, "hyps")) if unseen else None,
    }
    # Writes test_predictions.csv and metrics.json (the shared format used by all models).
    metrics = save_results(f"crnn_{a.subset}_greedy", res["paths"], res["refs"], res["hyps"], extra)
    print({k: metrics[k] for k in ("run_name", "num_samples", "cer", "wer", "accuracy", "params",
                                   "train_time_min", "ms_per_image", "cer_seen_words", "cer_unseen_words")})


if __name__ == "__main__":
    main()
