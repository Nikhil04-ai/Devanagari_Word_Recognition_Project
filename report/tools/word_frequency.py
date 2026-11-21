"""Table for report Section 7.4: test error grouped by how often the word occurs in training.

Every test word also occurs in the training split, so we cannot report "unseen words".
Instead the test images are grouped by how many times their word appears in train.csv.
All error rates come from the shared metric code (src/common/metrics.py).

Usage (from the repository root):
    python report/tools/word_frequency.py                 # Markdown table for every word-model run
    python report/tools/word_frequency.py --distribution  # how often test words occur in training

Runs are found automatically: every results/<run>/test_predictions.csv whose metrics.json says
model "crnn" or "trocr". The character CNN is skipped (it has no words).

Why this table exists: it shows whether the models do worse on words they saw rarely in training.
It is a report helper only. It does not train or test any model; it reads files that evaluation
already wrote (results/<run>/test_predictions.csv and metrics.json).

How it works:
1. Count how often every word appears in the training CSV.
2. For each test image, look up how often its true word appeared in training, and put the image
   in one of the groups in BUCKETS.
3. For each group and each run, compute CER and word accuracy with the shared metric code and
   print one Markdown table row per group.

"""
# Standard library: argparse = command-line flags, collections = Counter and defaultdict,
# csv / json = read the data files, glob = find files by pattern, os = paths,
# sys = change the import path, unicodedata = Unicode (NFC) normalisation.
import argparse
import collections
import csv
import glob
import json
import os
import sys
import unicodedata

# Make the repository root importable so the shared metric code (src.common) is used, never a copy.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Shared metrics: cer = character error rate, accuracy = exact-match share.
# E402 is a style-checker note: this import comes after the sys.path line on purpose.
from src.common.metrics import accuracy, cer  # noqa: E402

# Training and test word lists. The paths are relative to the repository root, so run from there.
TRAIN_CSV = "data/processed/words/train.csv"
TEST_CSV = "data/processed/words/test.csv"
# Values of the "model" field in metrics.json that are word models (the character CNN is left out).
WORD_MODELS = ("crnn", "trocr")

# Same groups as report Section 7.4. Every test word occurs 5 to 18 times in training
# (run with --distribution), so the usual "rare word" group (1 to 4) is empty.
# Each group is (label, smallest count, largest count), both ends included.
# 10**9 as the upper end of the last group means "no upper limit".
BUCKETS = [("5 to 6", 5, 6), ("7 to 9", 7, 9), ("10 or more", 10, 10**9)]


def norm(s):
    """Return s as text, stripped of outer spaces and normalised to Unicode NFC.
    Why: the same Devanagari word can be stored with different code point orders. After NFC they
    compare equal. This is the same step as _norm in src/common/metrics.py.
    """
    return unicodedata.normalize("NFC", str(s).strip())


def train_counts():
    """Count how often each word appears in the training CSV.
    Returns a Counter: word -> number of training images with that word.
    A word that never appears gives 0 (Counter does not raise an error).
    """
    # Read the CSV rows as dictionaries (the header names are the keys). "text" holds the word.
    with open(TRAIN_CSV, encoding="utf-8") as f:
        return collections.Counter(norm(r["text"]) for r in csv.DictReader(f))


def word_model_runs():
    """Find every finished word-model run in results/.
    Returns a list of (run_name, path of its test_predictions.csv).
    """
    runs = []
    # sorted() gives the same run order every time.
    for mpath in sorted(glob.glob("results/*/metrics.json")):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        # The predictions file is in the same folder as metrics.json.
        pred = os.path.join(os.path.dirname(mpath), "test_predictions.csv")
        # Keep the run only if it is a word model and its predictions file exists.
        if m.get("model") in WORD_MODELS and os.path.exists(pred):
            runs.append((m["run_name"], pred))
    return runs


def bucket_of(n):
    """Return the label of the group that contains count n, or None if no group fits."""
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


def show_distribution(counts):
    """Print how many test images have a word that occurs k times in training (for each k).
    Input: counts = the Counter from train_counts(). Prints a small text table, returns nothing.
    """
    # Read all test words (normalised).
    with open(TEST_CSV, encoding="utf-8") as f:
        test = [norm(r["text"]) for r in csv.DictReader(f)]
    # For every test word take its training count; then count how many test images share each count.
    dist = collections.Counter(counts[w] for w in test)
    # Number of test images and number of different words in training.
    print(f"{len(test)} test images; {len(counts)} distinct training words")
    print("occurrences in train | test images | share")
    for k in sorted(dist):
        # k and the counts are right-aligned in fixed-width columns; the share is a percent with 1 decimal.
        print(f"{k:>20} | {dist[k]:>11} | {100 * dist[k] / len(test):5.1f} %")


def table(counts):
    """Print the Markdown table: one row per group, with CER and word accuracy for every run.
    Input: counts = the Counter from train_counts(). Returns nothing.
    """
    runs = word_model_runs()
    if not runs:
        print("No word-model predictions found in results/.")
        return
    groups = {}  # run -> bucket -> (refs, hyps)
    # How many test images fall in each group. Filled once (for the first run) because every run
    # is tested on the same test images.
    sizes = collections.Counter()
    # Total number of test images (set from the first run, then kept).
    n_total = None
    for run, path in runs:
        # Read all rows of this run's predictions (columns: image_path, ground_truth, prediction).
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # n_total keeps its first value; later runs do not change it.
        n_total = n_total or len(rows)
        # defaultdict: a new group automatically starts as a pair of empty lists (refs, hyps).
        g = collections.defaultdict(lambda: ([], []))
        for r in rows:
            # Group of this image = group of the training count of its true word.
            b = bucket_of(counts[norm(r["ground_truth"])])
            # If a count fits no group, stop with a message. The groups in BUCKETS must be changed.
            if b is None:
                raise SystemExit(f"{run}: a test word occurs {counts[norm(r['ground_truth'])]} times in training; "
                                 "adjust BUCKETS")
            # Collect true words and predicted words of this group, so CER can be computed per group.
            g[b][0].append(r["ground_truth"])
            g[b][1].append(r["prediction"])
            # Count group sizes only for the first run (all runs have the same test images).
            if run == runs[0][0]:
                sizes[b] += 1
        groups[run] = g

    # Header row: two fixed columns, then CER and word accuracy columns for every run.
    head = "| Occurrences of the word in training | Share of test images | " + \
           " | ".join(f"CER {run} | Word acc. {run}" for run, _ in runs) + " |"
    print(head)
    # Markdown separator row. It has one "---" cell for each column (2 fixed + 2 per run).
    print("|" + "---|" * (2 + 2 * len(runs)))
    for name, _, _ in BUCKETS:
        # Row cells: group label and its share of all test images.
        cells = [name, f"{100 * sizes[name] / n_total:.1f} %"]
        for run, _ in runs:
            refs, hyps = groups[run][name]
            # Per-group CER and exact-match accuracy of this run, as percents with 2 decimals.
            cells += [f"{100 * cer(refs, hyps):.2f} %", f"{100 * accuracy(refs, hyps):.2f} %"]
        print("| " + " | ".join(cells) + " |")


def main():
    """Command-line entry point. Without a flag it prints the group table; with --distribution
    it prints how often test words occur in training.
    """
    # The module docstring is the help text; RawDescriptionHelpFormatter keeps its line breaks.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distribution", action="store_true", help="print how often test words occur in training")
    a = ap.parse_args()
    counts = train_counts()
    # A conditional expression used as a statement: run one function or the other, depending on the flag.
    show_distribution(counts) if a.distribution else table(counts)


# Run main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
