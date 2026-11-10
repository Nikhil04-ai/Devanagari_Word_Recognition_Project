"""Evaluate a trained CRNN on the TEST split once and save results with the shared writer.

    python -m src.crnn.evaluate --ckpt checkpoints/crnn_full/best.pt

Run from the repository root. Writes results/crnn_full_greedy/{test_predictions.csv, metrics.json}.
"""
import argparse
import csv
import os
import time

import torch
from PIL import Image
from torch import nn

from src.common.io import save_results
from src.common.metrics import cer

from .dataset import CharVocab, WordDataset, collate_fn, load_image
from .model import CRNN, count_params
from .train import evaluate_loader


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    vocab = CharVocab(ck["chars"])
    cfg = ck["cfg"]
    model = CRNN(len(vocab), hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab, ck


@torch.no_grad()
def ms_per_image(model, paths, root, device, invert, n=200, warmup=10):
    """Average time to run the network on one image (batch size 1)."""
    imgs = [load_image(Image.open(os.path.join(root, p)), invert=invert).unsqueeze(0).to(device)
            for p in paths[: n + warmup]]
    for x in imgs[:warmup]:
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x in imgs[warmup:]:
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / max(len(imgs) - warmup, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/processed/words")
    ap.add_argument("--csv", default="test.csv")
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only use the first N test images (quick check)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, ck = load_checkpoint(a.ckpt, device)
    invert = ck["cfg"].get("invert", False)

    ds = WordDataset(os.path.join(a.data_dir, a.csv), vocab, root=a.image_root, invert=invert)
    if a.limit:
        ds.samples = ds.samples[: a.limit]
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.num_workers, collate_fn=collate_fn)
    res = evaluate_loader(model, loader, vocab, device, nn.CTCLoss(blank=0, zero_infinity=True))

    # the dataset has few distinct words: also report test words seen / unseen in training
    with open(os.path.join(a.data_dir, "train.csv"), encoding="utf-8", newline="") as f:
        train_words = {r["text"] for r in csv.DictReader(f)}
    seen = [i for i, r in enumerate(res["refs"]) if r in train_words]
    unseen = [i for i in range(len(res["refs"])) if i not in set(seen)]
    pick = lambda idx, key: [res[key][i] for i in idx]  # noqa: E731
    extra = {
        "model": "crnn", "train_subset": a.subset, "decoding": "greedy",
        "params": count_params(model), "train_time_min": round(ck.get("train_time_min", 0.0), 2),
        "ms_per_image": round(ms_per_image(model, res["paths"], a.image_root, device, invert), 3),
        "best_epoch": ck["epoch"], "n_test_seen_words": len(seen), "n_test_unseen_words": len(unseen),
        "cer_seen_words": cer(pick(seen, "refs"), pick(seen, "hyps")) if seen else None,
        "cer_unseen_words": cer(pick(unseen, "refs"), pick(unseen, "hyps")) if unseen else None,
    }
    metrics = save_results(f"crnn_{a.subset}_greedy", res["paths"], res["refs"], res["hyps"], extra)
    print({k: metrics[k] for k in ("run_name", "num_samples", "cer", "wer", "accuracy", "params",
                                   "train_time_min", "ms_per_image", "cer_seen_words", "cer_unseen_words")})


if __name__ == "__main__":
    main()
