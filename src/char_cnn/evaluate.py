"""Evaluate the trained character CNN on the DHCD TEST split once and save the results.

    python -m src.char_cnn.evaluate --ckpt checkpoints/char_cnn_full/best.pt

Run from the repository root. Writes, in results/char_cnn_full_argmax/:
    test_predictions.csv, metrics.json        (shared writer, Contract 4)
    per_class.csv                             precision / recall / F1 / support per character
    confused_pairs.csv, confused_pairs.png    the 10 most frequent (true -> predicted) mistakes
    confusion_matrix.png                      46x46, rows normalised
    learning_curve.png                        from the training log (if log.csv is next to the checkpoint)

Where this file fits in the pipeline:
    train.py makes best.pt -> evaluate.py (this file) measures it on the Test split, which the model
    never saw during training or model choice, so the accuracy is an honest estimate.
    Run it once at the end, not after every change, so the Test split stays a fair final check.
"""
# argparse: command-line options. csv: result tables. os: paths. time: speed measurement.
import argparse
import csv
import os
import time

# numpy: confusion-matrix maths. torch: run the model.
import numpy as np
import torch

# save_results is the project's shared writer for predictions and metrics, so all models
# write the same file format.
from src.common.io import save_results

from .dataset import CHARS, CLASSES, NUM_CLASSES, load_split, to_input
from .model import CharCNN, count_params
# Reuse the validation function from train.py instead of writing a second one.
from .train import evaluate

# Default name of the results folder: <model>_<subset>_<decoding> (project naming convention).
RUN_NAME = "char_cnn_full_argmax"


def load_checkpoint(path, device):
    """Rebuild the model from a checkpoint file.

    path: .pt file made by train.py. device: torch.device to put the model on.
    Returns (model in eval mode, the full checkpoint dict).
    """
    # weights_only=False because the checkpoint also has plain Python objects (cfg, chars).
    ck = torch.load(path, map_location=device, weights_only=False)
    # The architecture must match the saved weights, so use the dropout from the saved config
    # (0.4 if an old checkpoint has none). Dropout has no weights, but we keep the same setup.
    model = CharCNN(NUM_CLASSES, dropout=ck["cfg"].get("dropout", 0.4)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()  # dropout off, BatchNorm uses its running statistics
    return model, ck


def confusion(y_true, y_pred, n=NUM_CLASSES):
    """Confusion matrix: cm[i, j] = number of images with true class i predicted as class j.

    y_true, y_pred: integer numpy arrays of the same length. n: number of classes.
    Returns an (n, n) int64 array. The diagonal holds the correct predictions.
    """
    cm = np.zeros((n, n), dtype=np.int64)
    # add.at adds 1 at every (true, pred) pair, and counts repeated pairs correctly
    # (plain cm[y_true, y_pred] += 1 would count a repeated pair only once).
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_scores(cm):
    """Precision, recall, F1 and support per class from a confusion matrix (rows = true).

    Input: cm (n, n). Returns four arrays of length n.
    precision = of the images predicted as this class, how many were right.
    recall    = of the images that truly are this class, how many were found.
    F1        = harmonic mean of precision and recall.
    support   = number of true images of the class.
    """
    tp = np.diag(cm).astype(float)  # true positives: correct predictions per class
    support = cm.sum(1)  # row sums = true images per class
    predicted = cm.sum(0)  # column sums = images predicted as each class
    # np.divide with `where` gives 0 instead of a divide-by-zero warning when a class has no predictions / no samples.
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def top_confusions(cm, k=10):
    """The k most frequent mistakes as a list of (true_class, predicted_class, count).

    Only real mistakes count (the diagonal is removed), and pairs with count 0 are dropped.
    """
    off = cm.copy()
    np.fill_diagonal(off, 0)  # remove the correct predictions, keep only mistakes
    # argsort on the flattened matrix (axis=None), reversed = biggest first, take k.
    order = np.argsort(off, axis=None)[::-1][:k]
    # unravel_index turns flat positions back into (row, column) = (true, predicted).
    return [(int(i), int(j), int(off[i, j])) for i, j in zip(*np.unravel_index(order, off.shape)) if off[i, j] > 0]


@torch.no_grad()
def ms_per_image(model, x_uint8, device, n=200, warmup=10):
    """Average time to classify one image (batch size 1).

    n: number of images timed. warmup: extra first runs that are not timed.
    Returns milliseconds per image (float).
    Why warmup: the first runs are slower (GPU start-up, caches), which would distort the average.
    """
    # Prepare all inputs first so data preparation is not part of the timing.
    xs = [to_input(x_uint8[i:i + 1].to(device)) for i in range(min(n + warmup, len(x_uint8)))]
    for x in xs[:warmup]:
        model(x)
    # GPU work is asynchronous: synchronize so the clock starts / stops when the GPU is really done.
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
    """Devanagari tick labels if a Devanagari font is installed, otherwise the romanised folder names.

    Returns (list of 46 labels, font name or None).
    Why: matplotlib's default font cannot draw Devanagari, so we need an installed font that can.
    """
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}  # names of all fonts matplotlib can see
    for name in ("Nirmala UI", "Noto Sans Devanagari", "Lohit Devanagari", "Mangal"):
        if name in installed:
            return CHARS, name
    # Fallback: the part of the folder name after the number, e.g. "character_1_ka" -> "ka".
    return [c.split("_", 2)[-1] for c in CLASSES], None


def save_figures(out, cm, pairs, x_test, y_test, log_csv):
    """Draw and save three figures into the folder `out`.

    cm: confusion matrix (46,46). pairs: output of top_confusions.
    x_test: dict {"images": uint8 array (N,32,32), "pred": predicted class array (N,)}.
    y_test: true class array (N,). log_csv: path of the training log (may not exist).
    """
    import matplotlib

    matplotlib.use("Agg")  # draw to files only, no window (works on servers and Kaggle)
    import matplotlib.pyplot as plt

    labels, font = plot_labels()
    if font:
        plt.rcParams["font.family"] = font

    # 1. confusion matrix, each row divided by its number of test images
    # Row normalisation: each row sums to 1, so a cell shows the share of that true class (classes are comparable).
    # np.maximum(..., 1) avoids dividing by zero for an empty row.
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
    plt.close(fig)  # free memory

    # 2. the most confused pairs: 3 test images of the true class that were predicted wrongly
    if pairs:
        # One row per confused pair, 3 example images per row.
        fig, axes = plt.subplots(len(pairs), 3, figsize=(4.5, 1.6 * len(pairs)), squeeze=False)
        for row, (t, p, n) in enumerate(pairs):
            # Test images whose true class is t but the model said p; keep at most the first 3.
            wrong = np.flatnonzero((y_test == t) & (x_test["pred"] == p))[:3]
            for col in range(3):
                a = axes[row, col]
                a.axis("off")
                if col < len(wrong):
                    a.imshow(x_test["images"][wrong[col]], cmap="gray")
            # Title shows: true -> predicted (number of times).
            axes[row, 0].set_title(f"{labels[t]} → {labels[p]}  ({n}×)", fontsize=9, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "confused_pairs.png"), dpi=120)
        plt.close(fig)

    # 3. learning curve
    # Only drawn if train.py left a log.csv next to the checkpoint.
    if log_csv and os.path.exists(log_csv):
        with open(log_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
        # Left plot: loss. A growing gap between train and validation loss means overfitting.
        a1.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
        a1.plot(ep, [float(r["val_loss"]) for r in rows], label="validation")
        a1.set_xlabel("epoch")
        a1.set_ylabel("loss")
        a1.legend()
        # Right plot: accuracy in %. Train accuracy is measured on augmented images, hence the label.
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
    """Command-line entry point: load the checkpoint, test it, write all result files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/char_cnn_full/best.pt")
    ap.add_argument("--data_dir", help="defaults to the data_dir stored in the checkpoint")
    ap.add_argument("--run_name", default=RUN_NAME)
    ap.add_argument("--no_figures", action="store_true")  # skip the plots (faster)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.ckpt, device)
    data_dir = a.data_dir or ck["cfg"]["data_dir"]

    # The official Test split: 300 images per class, never used for training or choosing the epoch.
    x_test, y_test, paths = load_split(data_dir, "Test")
    res = evaluate(model, x_test, y_test, device, torch.nn.CrossEntropyLoss())
    y_true, y_pred = y_test.numpy(), res["preds"].numpy()

    cm = confusion(y_true, y_pred)
    precision, recall, f1, support = per_class_scores(cm)
    pairs = top_confusions(cm, k=10)

    # Extra fields stored in metrics.json next to the standard ones that save_results computes.
    # Macro F1 = plain average of the 46 per-class F1 scores (every class counts the same).
    extra = {
        "model": "char_cnn", "train_subset": "full", "decoding": "argmax",
        "params": count_params(model), "train_time_min": round(ck["train_time_min"], 2),
        "ms_per_image": round(ms_per_image(model, x_test, device), 3),
        "macro_f1": float(f1.mean()), "best_epoch": ck["epoch"], "val_accuracy": ck["val_acc"],
        "note": "accuracy is the main metric; cer/wer are over Unicode code points of the single "
                "character (क्ष, त्र, ज्ञ have 3 code points), so cer is not exactly 1 - accuracy",
    }
    # Convert class indices to Devanagari strings so the shared writer can compute text metrics (CER / WER).
    refs = [CHARS[i] for i in y_true]
    hyps = [CHARS[i] for i in y_pred]
    metrics = save_results(a.run_name, paths, refs, hyps, extra)

    out = os.path.join("results", a.run_name)
    # newline="" is the csv module's rule: it stops blank lines appearing between rows on Windows.
    with open(os.path.join(out, "per_class.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "char", "precision", "recall", "f1", "support"])
        for i in range(NUM_CLASSES):
            w.writerow([CLASSES[i], CHARS[i], f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", support[i]])
    with open(os.path.join(out, "confused_pairs.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count", "share_of_true_class"])
        for t, p, n in pairs:
            # share = this mistake's count divided by the number of true images of that class.
            w.writerow([CHARS[t], CHARS[p], n, f"{n / max(support[t], 1):.4f}"])

    if not a.no_figures:
        # The training log is expected in the same folder as the checkpoint.
        log_csv = os.path.join(os.path.dirname(a.ckpt), "log.csv")
        save_figures(out, cm, pairs, {"images": x_test.numpy(), "pred": y_pred}, y_true, log_csv)

    # Final summary on screen.
    print(f"test accuracy {metrics['accuracy']:.4f} | macro F1 {extra['macro_f1']:.4f} | "
          f"{extra['params']:,} params | {extra['ms_per_image']} ms/image")
    print("most confused (true -> predicted, count):")
    for t, p, n in pairs:
        print(f"   {CHARS[t]} -> {CHARS[p]}  {n}")
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
