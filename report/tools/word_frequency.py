"""Table for report Section 7.4: test error grouped by how often the word occurs in training.

Every test word also occurs in the training split, so we cannot report "unseen words".
Instead the test images are grouped by how many times their word appears in train.csv.
All error rates come from the shared metric code (src/common/metrics.py).

Usage (from the repository root):
    python report/tools/word_frequency.py                 # Markdown table for every word-model run
    python report/tools/word_frequency.py --distribution  # how often test words occur in training

Runs are found automatically: every results/<run>/test_predictions.csv whose metrics.json says
model "crnn" or "trocr". The character CNN is skipped (it has no words).
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.metrics import accuracy, cer  # noqa: E402

TRAIN_CSV = "data/processed/words/train.csv"
TEST_CSV = "data/processed/words/test.csv"
WORD_MODELS = ("crnn", "trocr")

# Same groups as report Section 7.4. Every test word occurs 5 to 18 times in training
# (run with --distribution), so the usual "rare word" group (1 to 4) is empty.
BUCKETS = [("5 to 6", 5, 6), ("7 to 9", 7, 9), ("10 or more", 10, 10**9)]


def norm(s):
    return unicodedata.normalize("NFC", str(s).strip())


def train_counts():
    with open(TRAIN_CSV, encoding="utf-8") as f:
        return collections.Counter(norm(r["text"]) for r in csv.DictReader(f))


def word_model_runs():
    runs = []
    for mpath in sorted(glob.glob("results/*/metrics.json")):
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        pred = os.path.join(os.path.dirname(mpath), "test_predictions.csv")
        if m.get("model") in WORD_MODELS and os.path.exists(pred):
            runs.append((m["run_name"], pred))
    return runs


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


def show_distribution(counts):
    with open(TEST_CSV, encoding="utf-8") as f:
        test = [norm(r["text"]) for r in csv.DictReader(f)]
    dist = collections.Counter(counts[w] for w in test)
    print(f"{len(test)} test images; {len(counts)} distinct training words")
    print("occurrences in train | test images | share")
    for k in sorted(dist):
        print(f"{k:>20} | {dist[k]:>11} | {100 * dist[k] / len(test):5.1f} %")


def table(counts):
    runs = word_model_runs()
    if not runs:
        print("No word-model predictions found in results/.")
        return
    groups = {}  # run -> bucket -> (refs, hyps)
    sizes = collections.Counter()
    n_total = None
    for run, path in runs:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_total = n_total or len(rows)
        g = collections.defaultdict(lambda: ([], []))
        for r in rows:
            b = bucket_of(counts[norm(r["ground_truth"])])
            if b is None:
                raise SystemExit(f"{run}: a test word occurs {counts[norm(r['ground_truth'])]} times in training; "
                                 "adjust BUCKETS")
            g[b][0].append(r["ground_truth"])
            g[b][1].append(r["prediction"])
            if run == runs[0][0]:
                sizes[b] += 1
        groups[run] = g

    head = "| Occurrences of the word in training | Share of test images | " + \
           " | ".join(f"CER {run} | Word acc. {run}" for run, _ in runs) + " |"
    print(head)
    print("|" + "---|" * (2 + 2 * len(runs)))
    for name, _, _ in BUCKETS:
        cells = [name, f"{100 * sizes[name] / n_total:.1f} %"]
        for run, _ in runs:
            refs, hyps = groups[run][name]
            cells += [f"{100 * cer(refs, hyps):.2f} %", f"{100 * accuracy(refs, hyps):.2f} %"]
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distribution", action="store_true", help="print how often test words occur in training")
    a = ap.parse_args()
    counts = train_counts()
    show_distribution(counts) if a.distribution else table(counts)


if __name__ == "__main__":
    main()
