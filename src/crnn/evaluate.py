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

# ===== CONCEPTS USED IN THIS FILE =====
# Checkpoint (best.pt): the saved file from train.py with the weights, the character list and the config.
# Test set: data used only once at the end, after all choices were made on the validation set, so the score is honest.
# CER / WER / accuracy: error rates from metrics.py, written by save_results. CER = character edits / characters.
# CTC loss and blank: needed only because evaluate_loader also computes a loss; blank=0 is CharVocab.BLANK.
# Greedy decoding: best class per time step, merge repeats, drop blanks (decode.py, called in evaluate_loader).
# eval mode and torch.no_grad(): dropout off, BatchNorm uses stored statistics, no gradients are kept.
# state_dict / load_state_dict: the weights dictionary that is saved and loaded back into a CRNN of the same shape.
# Params: the count of trainable weights (count_params), a measure of model size.
# ms per image: the average time of one forward pass at batch size 1, a measure of speed.
# Warm-up and cuda.synchronize: the first runs are slow, and GPU work is asynchronous, so both are handled
#    before reading the clock in ms_per_image.
# Seen / unseen words: test words that also appear in the training labels, or not. The dataset has few distinct
#    words, so CER is also reported for the two groups.
# CSV: train.csv is read to get the set of training words; test_predictions.csv and metrics.json are written.
# argparse: reads the command-line flags in main().

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: The final test of the CRNN: it scores best.pt on the test split and saves the results in the shared format.
# Q: How is it run?
# A: From the repo root: python -m src.crnn.evaluate --ckpt checkpoints/crnn_full/best.pt
# Q: What does it read and write?
# A: It reads the checkpoint, test.csv, train.csv and the images. It writes results/crnn_<subset>_greedy/
#    test_predictions.csv and metrics.json (through save_results) and prints a summary.
# Q: Why is the test set used only here?
# A: Epochs and the best checkpoint were chosen on the validation set. A test set used once afterwards gives a
#    fair estimate of how the model works on new data.


def load_checkpoint(path, device):
    """Load a checkpoint written by train.py.

    path: .pt file. device: torch device to place the model on.
    Returns (model in eval mode, CharVocab, the raw checkpoint dict).
    """
    # Q: What does it do and why does it exist?
    # A: It loads best.pt, rebuilds the CharVocab and an identical CRNN, loads the saved weights and puts the
    #    model in eval mode. Testing needs the same network and the same character indices as in training.
    # Q: Why is the vocab built from ck["chars"] and why weights_only=False?
    # A: The saved character list guarantees the same indices as in training, even if vocab.txt changed later.
    #    The file also holds plain Python objects (cfg, chars), so weights_only=False is needed; load only your own.
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
    # Q: What does it do and why does it exist?
    # A: It loads up to n + warmup test images, runs the model on each one alone (batch size 1) and returns the
    #    average milliseconds per image. It exists to report the speed of the model next to CER / WER.
    # Q: Why the warm-up runs and torch.cuda.synchronize(), and what goes wrong without them?
    # A: The first calls are slower (start-up), and GPU work is asynchronous. Without warm-up and synchronize the
    #    clock would count start-up time or stop before the GPU finished, so the ms value would be wrong.
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
    # Q: What does it do and why does it exist?
    # A: It reads the flags, loads best.pt, runs the test split through evaluate_loader, adds model size, time,
    #    speed and seen / unseen CER, saves everything with save_results and prints a summary. It is the script entry.
    # Q: Why is CER also reported separately for seen and unseen test words?
    # A: The dataset has few distinct words. A model may do well only by remembering training words, so the two
    #    groups show how it works on words it has never seen. An empty group gives None, not a fake number.
    ap =argparse.ArgumentParser()
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
