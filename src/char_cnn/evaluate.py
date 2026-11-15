"""Evaluate the trained character CNN on the DHCD TEST split once and save the results.

    python -m src.char_cnn.evaluate --ckpt checkpoints/char_cnn_full/best.pt

Run from the repository root. Writes, in results/char_cnn_full_argmax/:
    test_predictions.csv, metrics.json        (shared writer, Contract 4)
    per_class.csv                             precision / recall / F1 / support per character
    confused_pairs.csv, confused_pairs.png    the 10 most frequent (true -> predicted) mistakes
    confusion_matrix.png                      46x46, rows normalised
    learning_curve.png                        from the training log (if log.csv is next to the checkpoint)
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from src.common.io import save_results

from .dataset import CHARS, CLASSES, NUM_CLASSES, load_split, to_input
from .model import CharCNN, count_params
from .train import evaluate

RUN_NAME = "char_cnn_full_argmax"


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = CharCNN(NUM_CLASSES, dropout=ck["cfg"].get("dropout", 0.4)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def confusion(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_scores(cm):
    """Precision, recall, F1 and support per class from a confusion matrix (rows = true)."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def top_confusions(cm, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    return [(int(i), int(j), int(off[i, j])) for i, j in zip(*np.unravel_index(order, off.shape)) if off[i, j] > 0]


@torch.no_grad()
def ms_per_image(model, x_uint8, device, n=200, warmup=10):
    """Average time to classify one image (batch size 1)."""
    xs = [to_input(x_uint8[i:i + 1].to(device)) for i in range(min(n + warmup, len(x_uint8)))]
    for x in xs[:warmup]:
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for x in xs[warmup:]:
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / max(len(xs) - warmup, 1)


# ---------------------------------------------------------------- figures
def plot_labels():
    """Devanagari tick labels if a Devanagari font is installed, otherwise the romanised folder names."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Nirmala UI", "Noto Sans Devanagari", "Lohit Devanagari", "Mangal"):
        if name in installed:
            return CHARS, name
    return [c.split("_", 2)[-1] for c in CLASSES], None


def save_figures(out, cm, pairs, x_test, y_test, log_csv):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, font = plot_labels()
    if font:
        plt.rcParams["font.family"] = font

    # 1. confusion matrix, each row divided by its number of test images
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), labels, fontsize=7, rotation=90)
    ax.set_yticks(range(NUM_CLASSES), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("DHCD test confusion matrix (rows normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # 2. the most confused pairs: 3 test images of the true class that were predicted wrongly
    if pairs:
        fig, axes = plt.subplots(len(pairs), 3, figsize=(4.5, 1.6 * len(pairs)), squeeze=False)
        for row, (t, p, n) in enumerate(pairs):
            wrong = np.flatnonzero((y_test == t) & (x_test["pred"] == p))[:3]
            for col in range(3):
                a = axes[row, col]
                a.axis("off")
                if col < len(wrong):
                    a.imshow(x_test["images"][wrong[col]], cmap="gray")
            axes[row, 0].set_title(f"{labels[t]} → {labels[p]}  ({n}×)", fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "confused_pairs.png"), dpi=120)
        plt.close(fig)

    # 3. learning curve
    if log_csv and os.path.exists(log_csv):
        with open(log_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
        a1.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
        a1.plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
        a1.set_xlabel("epoch")
        a1.set_ylabel("loss")
        a1.legend()
        a2.plot(ep, [100 * float(r["train_acc"]) for r in rows], label="train (augmented)")
        a2.plot(ep, [100 * float(r["val_acc"]) for r in rows], label="validation")
        a2.set_xlabel("epoch")
        a2.set_ylabel("accuracy (%)")
        a2.legend()
        fig.suptitle("Character CNN learning curve")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "learning_curve.png"), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/char_cnn_full/best.pt")
    ap.add_argument("--data_dir", help="defaults to the data_dir stored in the checkpoint")
    ap.add_argument("--run_name", default=RUN_NAME)
    ap.add_argument("--no_figures", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.ckpt, device)
    data_dir = a.data_dir or ck["cfg"]["data_dir"]

    x_test, y_test, paths = load_split(data_dir, "Test")
    res = evaluate(model, x_test, y_test, device, torch.nn.CrossEntropyLoss())
    y_true, y_pred = y_test.numpy(), res["preds"].numpy()

    cm = confusion(y_true, y_pred)
    precision, recall, f1, support = per_class_scores(cm)
    pairs = top_confusions(cm, k=10)

    extra = {
        "model": "char_cnn", "train_subset": "full", "decoding": "argmax",
        "params": count_params(model), "train_time_min": round(ck["train_time_min"], 2),
        "ms_per_image": round(ms_per_image(model, x_test, device), 3),
        "macro_f1": float(f1.mean()), "best_epoch": ck["epoch"], "val_accuracy": ck["val_acc"],
        "note": "accuracy is the main metric; cer/wer are over Unicode code points of the single "
                "character (क्ष, त्र, ज्ञ have 3 code points), so cer is not exactly 1 - accuracy",
    }
    refs = [CHARS[i] for i in y_true]
    hyps = [CHARS[i] for i in y_pred]
    metrics = save_results(a.run_name, paths, refs, hyps, extra)

    out = os.path.join("results", a.run_name)
    with open(os.path.join(out, "per_class.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "char", "precision", "recall", "f1", "support"])
        for i in range(NUM_CLASSES):
            w.writerow([CLASSES[i], CHARS[i], f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", support[i]])
    with open(os.path.join(out, "confused_pairs.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count", "share_of_true_class"])
        for t, p, n in pairs:
            w.writerow([CHARS[t], CHARS[p], n, f"{n / max(support[t], 1):.4f}"])

    if not a.no_figures:
        log_csv = os.path.join(os.path.dirname(a.ckpt), "log.csv")
        save_figures(out, cm, pairs, {"images": x_test.numpy(), "pred": y_pred}, y_true, log_csv)

    print(f"test accuracy {metrics['accuracy']:.4f} | macro F1 {extra['macro_f1']:.4f} | "
          f"{extra['params']:,} params | {extra['ms_per_image']} ms/image")
    print("most confused (true -> predicted, count):")
    for t, p, n in pairs:
        print(f"   {CHARS[t]} -> {CHARS[p]}  {n}")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
