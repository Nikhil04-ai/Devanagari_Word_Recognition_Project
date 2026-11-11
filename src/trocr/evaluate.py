"""Evaluate a trained TrOCR checkpoint on the TEST split once and save results with the shared writer.

    python -m src.trocr.evaluate --ckpt checkpoints/trocr_full/best.pt

Run from the repository root. Writes results/trocr_full_greedy/{test_predictions.csv, metrics.json}
(use --decoding beam for results/trocr_full_beam/).
"""
import argparse
import csv
import os
import time

import torch

from src.common.io import save_results
from src.common.metrics import cer
from src.common.seed import set_seed

from .config import load_config
from .dataset import TrOCRWordDataset, get_image_processor, make_loader
from .model import build_model, count_params, generate_ids
from .tokenizer import CharTokenizer
from .train import evaluate_loader


def load_checkpoint(path, device="cpu"):
    """Return (model, tokenizer, cfg, ckpt_dict) from a best.pt written by train.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = load_config(None, ck["config"])
    tok = CharTokenizer.from_dict(ck["tokenizer"])
    model = build_model(cfg, tok, load_pretrained=False)  # same architecture; weights come from the checkpoint
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), tok, cfg, ck


@torch.no_grad()
def time_per_image(model, ds, device, num_beams, max_length, n=200, warmup=10):
    """Average ms per image at batch size 1 (GPU if available), after warm-up."""
    px = [ds[i]["pixel_values"][None].to(device) for i in range(min(n + warmup, len(ds)))]
    warm = min(warmup, max(len(px) - 1, 0))
    for x in px[:warm]:
        generate_ids(model, x, num_beams=num_beams, max_length=max_length)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x in px[warm:]:
        generate_ids(model, x, num_beams=num_beams, max_length=max_length)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / max(len(px) - warm, 1)


def main():
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
    beams = a.num_beams if a.decoding == "beam" else 1

    ds = TrOCRWordDataset(os.path.join(a.data_dir, a.csv), tok, get_image_processor(cfg.pretrained),
                          max_length=cfg.max_length, root=a.image_root, first_n=a.limit)
    res = evaluate_loader(model, make_loader(ds, a.batch_size, False, a.num_workers), tok, cfg, device,
                          amp=device.type == "cuda", num_beams=beams, with_loss=False)
    ms = time_per_image(model, ds, device, beams, cfg.max_length)
    if device.type != "cuda":
        print("WARNING: no GPU found, ms_per_image was measured on CPU")

    # every test word also occurs in training (new writers, not new words): report seen / unseen anyway
    with open(os.path.join(a.data_dir, "train.csv"), encoding="utf-8", newline="") as f:
        train_words = {r["text"] for r in csv.DictReader(f)}
    seen = [i for i, r in enumerate(res["refs"]) if r in train_words]
    seen_set = set(seen)
    unseen = [i for i in range(len(res["refs"])) if i not in seen_set]
    pick = lambda idx, key: [res[key][i] for i in idx]  # noqa: E731
    extra = {
        "model": "trocr", "train_subset": a.subset, "decoding": a.decoding,
        "params": count_params(model, trainable_only=False),
        "train_time_min": round(float(ck.get("train_time_min", 0.0)), 2), "ms_per_image": round(ms, 3),
        "best_epoch": ck["epoch"], "n_test_seen_words": len(seen), "n_test_unseen_words": len(unseen),
        "cer_seen_words": cer(pick(seen, "refs"), pick(seen, "hyps")) if seen else None,
        "cer_unseen_words": cer(pick(unseen, "refs"), pick(unseen, "hyps")) if unseen else None,
    }
    m = save_results(f"trocr_{a.subset}_{a.decoding}", res["paths"], res["refs"], res["hyps"], extra)
    print({k: m[k] for k in ("run_name", "num_samples", "cer", "wer", "accuracy", "params",
                             "train_time_min", "ms_per_image")})


if __name__ == "__main__":
    main()
