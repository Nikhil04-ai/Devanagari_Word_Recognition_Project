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

# CONCEPTS USED IN THIS FILE
# - Test set: the official DHCD Test split (300 images per class). The model never saw it in training or when choosing
#   the epoch, so accuracy on it is an honest estimate. Loaded in main.
# - Accuracy: share of images classified correctly. Taken from the shared save_results writer.
# - Confusion matrix: a table where cell [i, j] counts images of true class i predicted as class j. Built by
#   confusion; the diagonal is the correct predictions.
# - Row normalisation: each row is divided by its total so every cell is a share of that true class, making classes
#   comparable. Used in the confusion_matrix.png plot.
# - Precision, recall, F1, support: precision = of the images predicted as a class, how many were right; recall = of
#   the true images of a class, how many were found; F1 = harmonic mean of the two; support = number of true images.
#   Computed in per_class_scores.
# - Macro-F1: the plain average of the 46 per-class F1 scores, so every class counts the same. Stored in metrics.json.
# - CER / WER: character and word error rates between reference and predicted text. They are computed inside
#   save_results from the strings built in main; here each text is one character.
# - Logits and argmax: the model gives 46 scores; the index of the largest is the predicted class (done inside
#   train.evaluate).
# - Checkpoint / state_dict: the saved .pt file with the weights and settings; load_checkpoint rebuilds the model from
#   it and sets eval mode.
# - Latency (ms per image): time to classify one image. ms_per_image times it after a warm-up, with cuda synchronize
#   on GPU because GPU work is asynchronous.
# - Learning curve and overfitting: train vs validation loss / accuracy per epoch, read from log.csv. A growing gap
#   means overfitting. Drawn in save_figures.
# - Unicode code points and fonts: some characters use several code points, so CER is not exactly 1 - accuracy.
#   matplotlib needs a Devanagari font to draw the labels (plot_labels).
# - matplotlib 'Agg' backend: draws straight to image files with no window, so it works on servers and Kaggle.

# FILE Q&A (viva)
# Q: What part of the project is this file?
# A: The final test of Model A: it measures the best checkpoint once on the Test split and writes metrics, tables and
#    figures.
# Q: How is it run?
# A: From the repo root: python -m src.char_cnn.evaluate --ckpt checkpoints/char_cnn_full/best.pt. Optional flags:
#    --data_dir, --run_name, --no_figures.
# Q: What does it read and write?
# A: It reads the checkpoint, the Test split, and log.csv next to the checkpoint. It writes into results/<run_name>/:
#    test_predictions.csv and metrics.json (save_results), per_class.csv, confused_pairs.csv, and the PNG figures.
# Q: Why run it only once?
# A: If you tune the model after looking at Test results, the Test split stops being a fair final check.

# Default name of the results folder: <model>_<subset>_<decoding> (project naming convention).
RUN_NAME = "char_cnn_full_argmax"


def load_checkpoint(path, device):
    """Rebuild the model from a checkpoint file.

    path: .pt file made by train.py. device: torch.device to put the model on.
    Returns (model in eval mode, the full checkpoint dict).
    """
    # Q: What does load_checkpoint do and why does it exist?
    # A: It rebuilds CharCNN from a .pt file made by train.py, loads the weights and sets eval mode. It returns the
    #    model and the full checkpoint dict.
    # Q: Why use the dropout from the saved config, and why weights_only=False?
    # A: load_state_dict needs the same architecture, so the saved setting is used (0.4 if an old checkpoint has
    #    none). weights_only=False because the file also holds plain Python objects (cfg, chars).
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
    # Q: What does confusion do and why does it exist?
    # A: It builds the n x n confusion matrix: cm[i, j] counts true class i predicted as j. It is the base for per-
    #    class scores, the top mistakes and the plot.
    # Q: Why np.add.at and not cm[y_true, y_pred] += 1?
    # A: With fancy indexing a repeated (true, pred) pair is counted only once. np.add.at adds 1 for every pair, so
    #    repeats are counted correctly.
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
    # Q: What does per_class_scores do and why does it exist?
    # A: It computes precision, recall, F1 and support for each class from the confusion matrix. The F1 values give
    #    the macro-F1 and the per_class.csv table.
    # Q: Why np.divide with where=, and why rows as true classes?
    # A: A class with no predictions or no samples would divide by zero; where= gives 0 instead. Rows are true
    #    classes, so row sums are the support and column sums are the predicted counts.
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
    # Q: What does top_confusions do and why does it exist?
    # A: It returns the k most frequent mistakes as (true, predicted, count). They show which characters the model
    #    mixes up.
    # Q: Why zero the diagonal and drop zero counts?
    # A: The diagonal holds correct predictions, and we only want mistakes. Pairs with count 0 are not mistakes, so
    #    they are removed from the list.
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
    # Q: What does ms_per_image do and why does it exist?
    # A: It measures the average time to classify one image (batch size 1). It is stored in metrics.json as the speed
    #    of the model.
    # Q: Why warm-up runs and cuda synchronize?
    # A: The first runs are slower (GPU start-up, caches) and would distort the average. GPU work is asynchronous, so
    #    synchronize makes the clock start and stop when the GPU is really finished.
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
    # Q: What does plot_labels do and why does it exist?
    # A: It returns the tick labels for the plots: Devanagari characters if a suitable font is installed, otherwise
    #    romanised folder names. It also returns the font name or None.
    # Q: Why is there a fallback?
    # A: matplotlib's default font cannot draw Devanagari. Without a fallback the labels would not show correctly, so
    #    the folder names (like 'ka') are used instead.
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
    # Q: What does save_figures do and why does it exist?
    # A: It draws and saves the confusion matrix, the picture of the most confused pairs, and the learning curve (only
    #    if log.csv exists). These are the report figures.
    # Q: Why the Agg backend, plt.close, and row normalisation?
    # A: Agg draws to files without a window, so it works on servers; plt.close frees memory after each figure. Rows
    #    are normalised so each cell is a share of its true class and classes are comparable.
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
    # Q: What does main do and why does it exist?
    # A: It is the command-line entry point: load the checkpoint, run the model on the Test split, compute metrics,
    #    write all result files and print a summary.
    # Q: Why convert class indices to characters before save_results?
    # A: save_results computes text metrics (CER / WER) from strings, so it needs Devanagari strings, not class
    #    numbers. Extra fields such as macro-F1 are added to metrics.json through `extra`.
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
