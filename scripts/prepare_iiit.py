"""Turn the raw IIIT-HW-Dev (Version 1) files into the CSVs and character list of CONTRACTS.md.

Raw layout (from the dataset's Readme.txt): train.txt / val.txt / test.txt, one line per image:
    HindiSeg/val/3/165/18.jpg <word>        (relative path, a space, the word in Unicode)

Run from the repository root:
    python scripts/prepare_iiit.py

Writes data/processed/words/{train,val,test}.csv (image_path,text), vocab.txt (one code point per
line, from the training split only) and stats.json (numbers for the report).

Where this file fits in the pipeline:
    download_iiit.py (archive) -> prepare_iiit.py (this file) -> src/crnn/dataset.py reads the CSVs
    and vocab.txt -> train.py.

What it does, step by step:
    1. Read the three raw list files.
    2. Normalise every label to NFC and drop bad rows (unreadable image, tiny image, empty label).
    3. Build the character list (vocab) from the TRAINING split only.
    4. Write the CSVs and vocab.txt.
    5. Compute statistics (word lengths, writers, unseen words, CTC fit ...) for the report.
"""
# Standard library: arguments, CSV / JSON output, paths, random sampling, statistics, Unicode, counting, threads.
import argparse
import csv
import json
import os
import random
import statistics
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# numpy + PIL: read images and measure them.
import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
IMG_H, MIN_W, MAX_W, DOWN = 64, 128, 512, 4  # must match src/crnn/dataset.py


def read_split(raw, name):
    """Read raw/<name>.txt. Returns a list of (relative image path, label) tuples.

    raw: folder with the raw files. name: "train", "val" or "test".
    """
    rows = []
    with open(os.path.join(raw, f"{name}.txt"), encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:  # skip empty lines
                # Split at the FIRST space only: before it is the path, everything after it is the label.
                path, _, label = line.partition(" ")
                rows.append((path, label))
    return rows


def image_size(path):
    """Return (width, height) of an image file, or None if it is missing or cannot be opened."""
    try:
        with Image.open(path) as im:
            return im.size  # reads only the header
    except Exception:
        return None  # the caller counts this as a "missing or unreadable image"


def min_ctc_steps(label):
    """CTC needs one time step per character plus one blank between equal neighbours.

    label: string. Returns an int. The same rule as min_ctc_input_length in src/crnn/dataset.py,
    but here it works on a string instead of a list of indices.
    """
    return len(label) + sum(1 for a, b in zip(label, label[1:]) if a == b)


def available_steps(w, h):
    """Number of CTC time steps the CRNN will have for an image of width w and height h (pixels).

    It repeats the width logic of load_image in dataset.py: scale to height IMG_H keeping the aspect
    ratio, cap at MAX_W, pad up to MIN_W, then divide by DOWN (the CNN shrinks the width by 4).
    """
    width = min(max(1, round(w * IMG_H / max(h, 1))), MAX_W)
    return max(width, MIN_W) // DOWN


def pct(xs, q):
    """Simple percentile: the value at position q (0 to 1) of the sorted list xs."""
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)]  # min(...) keeps the index inside the list when q = 1


def main():
    """Run all preparation steps and print a summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/iiit_hw_dev", help="folder with train.txt, val.txt, test.txt, HindiSeg/")
    ap.add_argument("--out", default="data/processed/words")
    ap.add_argument("--threads", type=int, default=16)  # threads that read image headers in parallel
    args = ap.parse_args()
    # Path prefix for the CSV: forward slashes (works on Windows and Linux), no trailing slash.
    raw_rel = args.raw.replace("\\", "/").rstrip("/")
    os.makedirs(args.out, exist_ok=True)

    # data[split] = kept rows; stats = numbers that are saved to stats.json
    data, stats = {}, {"dropped": {}}
    for split in SPLITS:
        rows = read_split(args.raw, split)
        full = [os.path.join(args.raw, p) for p, _ in rows]
        # Reading many image headers is slow on disk, but threads can wait for the disk in parallel.
        with ThreadPoolExecutor(args.threads) as ex:
            sizes = list(ex.map(image_size, full))  # same order as rows
        keep, dropped = [], Counter()  # dropped counts the rows removed, per reason
        nfc_changed = 0  # how many labels NFC normalisation actually changed
        for (path, label), size in zip(rows, sizes):
            # NFC: one standard code-point order for each Devanagari word. The model, the vocab and
            # the metrics all use NFC, so the labels must too.
            norm = unicodedata.normalize("NFC", label.strip())
            if norm != label:
                nfc_changed += 1
            if size is None:
                dropped["missing or unreadable image"] += 1
            elif size[0] < 2 or size[1] < 2:
                dropped["image smaller than 2 px"] += 1
            elif not norm:
                dropped["empty label"] += 1
            else:
                keep.append((path, norm, size))
        data[split] = keep
        stats["dropped"][split] = dict(dropped)
        stats[f"{split}_raw_lines"] = len(rows)
        stats[f"{split}_labels_changed_by_nfc"] = nfc_changed
        print(f"{split}: {len(rows)} lines, kept {len(keep)}, dropped {dict(dropped) or 0}, "
              f"labels changed by NFC: {nfc_changed}", flush=True)

    # character list from the training split only
    # Only training labels define the vocab, so nothing from val / test is used to build the model.
    # Characters that appear only in val / test are counted as out-of-vocabulary in the stats below.
    # sorted() makes vocab.txt the same on every run.
    vocab = sorted({c for _, lab, _ in data["train"] for c in lab})
    vset = set(vocab)  # a set is faster for "is this character in the vocab?" checks
    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(vocab) + "\n")
    for split in SPLITS:
        with open(os.path.join(args.out, f"{split}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")  # "\n" line ends on every OS
            w.writerow(["image_path", "text"])
            # Path relative to the repository root, as dataset.py expects (it joins it with image_root).
            w.writerows((f"{raw_rel}/{p}", lab) for p, lab, _ in data[split])

    # ---- statistics for the report and for the model settings
    train_words = {lab for _, lab, _ in data["train"]}  # all distinct training words
    stats["vocab_size_code_points"] = len(vocab)
    stats["hyphen_in_vocab"] = "-" in vset
    for split in SPLITS:
        rows = data[split]
        labels = [lab for _, lab, _ in rows]
        # The path is like HindiSeg/val/3/165/18.jpg: part 2 is the writer id here, part 3 the page.
        writers = Counter(p.split("/")[2] for p, _, _ in rows)
        lens = [len(x) for x in labels]
        # True for each image whose label can fit in the available CTC time steps.
        steps_ok = [min_ctc_steps(lab) <= available_steps(*sz) for _, lab, sz in rows]
        # Width after resizing to height 64 (capped at MAX_W), before any padding.
        widths64 = [min(round(w_ * IMG_H / h_), MAX_W) for _, _, (w_, h_) in rows]
        # Characters of this split that the training vocab does not contain, with their counts.
        oov = Counter(c for lab in labels for c in lab if c not in vset)
        stats[split] = {
            "images": len(rows),
            "distinct_words": len(set(labels)),
            "writers": len(writers),
            "pages": len({tuple(p.split("/")[2:4]) for p, _, _ in rows}),
            "label_len_median": statistics.median(lens),
            "label_len_p95": pct(lens, 0.95),
            "label_len_max": max(lens),
            # How many images get squeezed (too wide) or padded (too narrow) by the CRNN preprocessing.
            "images_wider_than_512_after_resize": sum(1 for p_, _, (w_, h_) in rows if round(w_ * IMG_H / h_) > MAX_W),
            "images_narrower_than_128_after_resize": sum(1 for x in widths64 if x < MIN_W),
            "resized_width_median": statistics.median(widths64),
            # Labels that cannot fit in the CTC time steps would give an infinite loss.
            "labels_too_long_for_ctc": steps_ok.count(False),
            "chars_not_in_train_vocab": dict(oov),
            "rows_with_char_not_in_train_vocab": sum(1 for lab in labels if any(c not in vset for c in lab)),
            # How much of this split is made of words the model saw in training (the vocabulary of words is small).
            "share_of_images_whose_word_is_in_train": round(sum(lab in train_words for lab in labels) / len(labels), 4),
            "distinct_words_not_in_train": len({lab for lab in labels if lab not in train_words}),
            "numeric_labels": sum(lab.isdigit() for lab in labels),
        }
        # Sort writer ids as numbers when possible (so 2 comes before 10), otherwise as text.
        stats[split]["writer_ids"] = sorted(writers, key=lambda k: int(k) if k.isdigit() else k)
    ws = {s: set(stats[s]["writer_ids"]) for s in SPLITS}
    # Writers that appear in two splits (an empty list means the splits are writer-independent).
    stats["writers_shared_train_val"] = sorted(ws["train"] & ws["val"])
    stats["writers_shared_train_test"] = sorted(ws["train"] & ws["test"])
    stats["writers_shared_val_test"] = sorted(ws["val"] & ws["test"])

    # background colour: is the paper light (dark ink on white)?
    # The CRNN code expects white paper (dataset.py makes white = +1.0). This check shows if the images match.
    random.seed(42)  # fixed seed: the same 300 images are sampled on every run
    sample = random.sample(data["train"], 300)
    light = 0
    for p, _, _ in sample:
        a = np.asarray(Image.open(os.path.join(args.raw, p)).convert("L"))
        # Most pixels are paper, so the median pixel value is the background. Above 127 = light.
        light += int(np.median(a) > 127)
    stats["share_of_300_train_samples_with_light_background"] = light / 300

    # ensure_ascii=False keeps any Devanagari text readable in the file.
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Short summary printed to the screen.
    print("\n=== summary ===")
    for split in SPLITS:
        s = stats[split]
        print(f"{split:5s} images {s['images']:6d} | distinct words {s['distinct_words']:5d} | writers {s['writers']:3d} | "
              f"label length median {s['label_len_median']} p95 {s['label_len_p95']} max {s['label_len_max']} | "
              f"words also in train {s['share_of_images_whose_word_is_in_train']:.1%}")
    print(f"characters in vocab.txt: {len(vocab)}")
    for split in ("val", "test"):
        s = stats[split]
        print(f"{split}: characters not in train vocab {s['chars_not_in_train_vocab']} "
              f"({s['rows_with_char_not_in_train_vocab']} rows) | labels too long for CTC: {s['labels_too_long_for_ctc']}")
    print(f"train: labels too long for CTC: {stats['train']['labels_too_long_for_ctc']}")
    print(f"writers shared train/val: {stats['writers_shared_train_val']}  train/test: {stats['writers_shared_train_test']}  "
          f"val/test: {stats['writers_shared_val_test']}")
    print(f"light background in {stats['share_of_300_train_samples_with_light_background']:.0%} of 300 sampled train images")
    print(f"wrote {args.out}/ (train.csv, val.csv, test.csv, vocab.txt, stats.json)")


if __name__ == "__main__":
    main()
