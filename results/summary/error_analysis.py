"""Short error analysis + results chart for every finished run (owner: Ravi).

    python results/summary/error_analysis.py

Run from the repository root. Reads every results/<run>/metrics.json and test_predictions.csv
(missing runs are simply skipped), and writes to results/summary/:

    results_chart.png        test CER and WER of every run (one bar group per run)
    cer_by_length.png        word models: CER by word length (in code points)
    error_analysis.md        tables for the report: error types, top character substitutions,
                             CER by word length, and the CNN's most confused characters

The word models are aligned with the same edit distance as src/common/metrics.py (code points),
so the counts add up to the CER numerator.

Where it fits: this is the last analysis step. Training and evaluation have already written the
numbers; this script only reads them and makes the tables and charts used in the report.

How it works, in short:
1. load_runs() reads every finished run (metrics.json and, if present, test_predictions.csv).
2. For the word models, analyse_words() lines up each true word with the predicted word using
   Levenshtein edit operations (substitution, deletion, insertion). Each error is counted by the
   type of the character involved (consonant, vowel sign, halant ...), by character pair, and by
   word length.
3. For the character CNN, the script lists the most confused characters instead.
4. main() writes the Markdown tables first, then the two charts with matplotlib.

"""
# Standard library: csv / json = read result files, glob = find files by pattern, os = paths,
# sys = exit with a message, unicodedata = NFC normalisation, Counter = counting.
import csv
import glob
import json
import os
import sys
import unicodedata
from collections import Counter

# rapidfuzz: fast edit distance. Levenshtein.editops gives the list of single edits.
from rapidfuzz.distance import Levenshtein

# Folder where the tables and charts are written.
OUT = os.path.join("results", "summary")
# Folders inside results/ that are not model runs and must be skipped.
SKIP = {"summary", "eda", "error_analysis"}

# Dependent vowel signs (matras). They are written after a consonant and change its vowel sound.
MATRAS = set("ािीुूृॄॅॆेैॉॊोौॢॣ")
# Punctuation: danda, double danda, hyphen, full stop, comma.
PUNCT = set("।॥-.,")
MARKS = set("ंँः")  # anusvara, chandrabindu, visarga
# Halant (virama) joins consonants into a conjunct. Nukta is the dot under a letter.
HALANT, NUKTA = "्", "़"


def category(ch):
    """Return the name of the character group that character ch belongs to.
    Input: one character. Returns a label used as a row name in the error table.
    The checks run in order, so the first match wins. Anything that matches no group is "other".
    """
    if ch in MATRAS:
        return "vowel sign (matra)"
    if ch in MARKS:
        return "anusvara / chandrabindu / visarga"
    if ch == HALANT:
        return "halant (conjuncts)"
    if ch == NUKTA:
        return "nukta"
    if ch in PUNCT:
        return "punctuation (danda, hyphen)"
    # Devanagari digits U+0966 to U+096F, or any other digit.
    if "०" <= ch <= "९" or ch.isdigit():
        return "digit"
    # Range of the independent vowels.
    if "अ" <= ch <= "औ":
        return "independent vowel"
    # Two consonant ranges: the normal consonants, and the second range after the or.
    if "क" <= ch <= "ह" or "क़" <= ch <= "य़":
        return "consonant"
    return "other"


def nfc(s):
    """Return s as text, stripped and in Unicode NFC form (None becomes an empty string).
    Why: the same word can be stored with different code point orders; NFC makes them equal.
    """
    return unicodedata.normalize("NFC", (s or "").strip())


def load_runs():
    """Read every finished run in results/.
    Returns a list of dicts: name (folder name), metrics (from metrics.json), rows (list of
    (true text, predicted text) pairs; empty if there is no predictions file).
    """
    runs = []
    # sorted() gives the same run order every time.
    for mpath in sorted(glob.glob(os.path.join("results", "*", "metrics.json"))):
        # Run name = name of the folder that holds metrics.json.
        name = os.path.basename(os.path.dirname(mpath))
        # Skip folders that are not model runs.
        if name in SKIP:
            continue
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        # The predictions file sits in the same folder as metrics.json.
        ppath = os.path.join(os.path.dirname(mpath), "test_predictions.csv")
        rows = []
        # Without a predictions file the run is still listed (rows stay empty).
        if os.path.exists(ppath):
            # newline="" is the correct way to open a file for the csv module.
            with open(ppath, encoding="utf-8", newline="") as f:
                rows = [(nfc(r["ground_truth"]), nfc(r["prediction"])) for r in csv.DictReader(f)]
        # Store the run as a dict.
        runs.append({"name": name, "metrics": m, "rows": rows})
    return runs


def analyse_words(rows):
    """Count edit operations per character category, substitution pairs, and CER by length.

    Input: rows = list of (true word, predicted word) pairs.
    Returns four things: ops (edit counts), subs (substitution pair counts), by_len (errors, characters
    and words per length group) and ref_chars (how many true characters of each type there are).
    """
    # Count of each kind of edit per character type.
    ops = Counter()             # (op, category) -> count, category of the TRUE char (insert: predicted char)
    subs = Counter()            # (true char, predicted char) -> count
    by_len = {}                 # length bucket -> [errors, chars]
    ref_chars = Counter()
    # Go through the word pairs one by one.
    for ref, hyp in rows:
        # Count the true characters by type (needed for the share of test characters).
        for ch in ref:
            ref_chars[category(ch)] += 1
        # Number of edits in this word.
        errors = 0
        # editops lists the single edits that turn the true word into the predicted word. Each edit has a
        # tag (replace, delete or insert) and positions: src_pos in the true word, dest_pos in the prediction.
        for op in Levenshtein.editops(ref, hyp):
            errors += 1
            # Replace: true character a was read as b.
            if op.tag == "replace":
                a, b = ref[op.src_pos], hyp[op.dest_pos]
                ops[("substitution", category(a))] += 1
                subs[(a, b)] += 1
            # Delete: a true character is missing in the prediction.
            elif op.tag == "delete":
                ops[("deletion", category(ref[op.src_pos]))] += 1
            # Insert: the prediction has an extra character (counted by the type of the extra character).
            else:
                ops[("insertion", category(hyp[op.dest_pos]))] += 1
        # Word length in code points (the same unit the CER uses).
        n = len(ref)
        # Put the word in a length group, using a chain of one-line if / else tests.
        bucket = "1-3" if n <= 3 else "4-5" if n <= 5 else "6-7" if n <= 7 else "8-10" if n <= 10 else "11+"
        # Per group: [errors, true characters, words]. setdefault creates it on first use.
        e = by_len.setdefault(bucket, [0, 0, 0])
        e[0] += errors
        e[1] += n
        e[2] += 1
    return ops, subs, by_len, ref_chars


# Order of the length groups in tables and in the chart.
BUCKETS = ["1-3", "4-5", "6-7", "8-10", "11+"]


def pct(x):
    """Format a fraction (0.05) as a percent text with 2 decimals ("5.00 %")."""
    return f"{100 * x:.2f} %"


def main():
    """Build all tables and charts. Reads the results folders, writes error_analysis.md,
    results_chart.png and (if a word model has predictions) cer_by_length.png.
    Stops with a message if no metrics.json file is found.
    """
    # Create results/summary if it does not exist yet.
    os.makedirs(OUT, exist_ok=True)
    runs = load_runs()
    # No runs means the script was started from the wrong folder or before any evaluation.
    if not runs:
        sys.exit("no results/*/metrics.json found: run from the repository root after at least one evaluation")
    # md is a list of lines of the Markdown file. They are joined at the end.
    md = ["# Error analysis (generated by results/summary/error_analysis.py)", ""]

    # ---- overview
    # Table header row and separator row for the overview table.
    md += ["## Test results", "", "| Run | Model | Level | CER | WER | Accuracy | Test images |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r["metrics"]
        # The character CNN works on single characters, so it is a "character" level model.
        level = "character" if m["model"] == "char_cnn" else "word"
        # WER has no meaning for single characters, so the CNN gets n/a.
        wer = "n/a" if m["model"] == "char_cnn" else pct(m["wer"])
        # One table row per run.
        md.append(f"| {r['name']} | {m['model']} | {level} | {pct(m['cer'])} | {wer} | {pct(m['accuracy'])} | {m['num_samples']} |")
    # Note under the table: the CNN and the word models solve different tasks.
    md += ["", "The CNN reads one isolated consonant or digit; the word models read whole words with vowel signs "
           "and conjuncts. The numbers measure different tasks and are not directly comparable.", ""]

    # Run name -> list of (length group, CER). Used for the line chart later.
    length_curves = {}
    for r in runs:
        m = r["metrics"]
        # Skip a run that has no predictions file.
        if not r["rows"]:
            continue
        # The character CNN gets a different analysis: which true characters are confused with which.
        if m["model"] == "char_cnn":
            # Count every (true, predicted) pair where the two differ.
            conf = Counter((t, p) for t, p in r["rows"] if t != p)
            # Number of test samples of each true class (used to turn counts into a share).
            per_class = Counter(t for t, _ in r["rows"])
            md += [f"## {r['name']}: most confused characters", "",
                   "| True | Predicted | Count | Share of the true class |", "|---|---|---|---|"]
            # Show the 10 most common confusions. The share is the count divided by the samples of the true class.
            for (t, p), n in conf.most_common(10):
                md.append(f"| {t} | {p} | {n} | {pct(n / per_class[t])} |")
            md.append("")
            # Done with this run. continue skips the word-model part below.
            continue

        # Word model: count edit types, substitution pairs and errors by length.
        ops, subs, by_len, ref_chars = analyse_words(r["rows"])
        # Total number of character errors (substitutions + deletions + insertions).
        total_err = sum(ops.values())
        # Heading and summary line. The {:,} format adds thousands separators.
        md += [f"## {r['name']}", "",
               f"{total_err:,} character errors on {len(r['rows']):,} test words "
               f"({sum(1 for a, b in r['rows'] if a != b):,} words wrong).", "",
               "### Errors by type of the true character", "",
               "| Character type | Share of test characters | Substitutions | Deletions | Insertions | Share of all errors |",
               "|---|---|---|---|---|---|"]
        # Character types, sorted so that the type with the most errors comes first.
        # The set joins the types seen in errors and the types seen in the true text.
        cats = sorted({c for _, c in ops} | set(ref_chars), key=lambda c: -sum(ops[(o, c)] for o in ("substitution", "deletion", "insertion")))
        # Total number of true characters (for the share column).
        n_chars = sum(ref_chars.values())
        for c in cats:
            # s, d, i = substitutions, deletions and insertions for this character type.
            s, d, i = ops[("substitution", c)], ops[("deletion", c)], ops[("insertion", c)]
            # Table row. max(..., 1) avoids a division by zero when the total is 0.
            md.append(f"| {c} | {pct(ref_chars[c] / max(n_chars, 1))} | {s} | {d} | {i} | {pct((s + d + i) / max(total_err, 1))} |")
        # Table of the 15 most frequent substitutions.
        md += ["", "### Top 15 character substitutions (true → predicted)", "", "| True | Predicted | Count |", "|---|---|---|"]
        for (a, b), n in subs.most_common(15):
            md.append(f"| {a} | {b} | {n} |")
        # Table of CER by word length.
        md += ["", "### CER by word length (code points)", "", "| Length | Test words | CER |", "|---|---|---|"]
        # Points for the line chart: (length group, CER of that group).
        curve = []
        for b in BUCKETS:
            if b in by_len:
                # e = errors, n = true characters, w = number of words in this group.
                e, n, w = by_len[b]
                # CER of the group = errors / true characters.
                md.append(f"| {b} | {w} | {pct(e / n)} |")
                curve.append((b, e / n))
        # Keep the points for the line chart.
        length_curves[r["name"]] = curve
        md.append("")

    # Write all Markdown lines to error_analysis.md.
    with open(os.path.join(OUT, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- figures (labels are Latin only, so no Devanagari font is needed)
    # matplotlib is imported here, after the Markdown file is written.
    import matplotlib

    # Agg backend: draws into image files and needs no window.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Data for the bar chart: run names, CER in percent, WER in percent.
    names = [r["name"] for r in runs]
    cers = [100 * r["metrics"]["cer"] for r in runs]
    # WER is set to 0 for the character CNN (it has no WER). The 0 bar has no label (see below).
    wers = [100 * r["metrics"]["wer"] if r["metrics"]["model"] != "char_cnn" else 0 for r in runs]
    # New figure. Width grows with the number of runs (at least 5 inches); height 3.8 inches.
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(runs)), 3.8))
    # x positions of the bar groups.
    xs = range(len(runs))
    # Two bars per run, one shifted left and one right by 0.2. Each bar is 0.4 wide.
    b1 = ax.bar([x - 0.2 for x in xs], cers, 0.4, label="CER")
    b2 = ax.bar([x + 0.2 for x in xs], wers, 0.4, label="WER (word models)")
    # Write the value on top of each bar.
    ax.bar_label(b1, fmt="%.1f", fontsize=8)
    # Empty label where the WER is 0, so the CNN shows no WER number.
    ax.bar_label(b2, labels=[f"{w:.1f}" if w else "" for w in wers], fontsize=8)
    # Run names under the bars, slightly rotated so they fit.
    ax.set_xticks(list(xs), names, rotation=15, fontsize=8)
    ax.set_ylabel("error on the test set (%)")
    ax.set_title("Test error per run\n(the character CNN and the word models measure different tasks)", fontsize=9)
    ax.legend(fontsize=8)
    # tight_layout stops labels from being cut off.
    fig.tight_layout()
    # Save the chart as a PNG at 150 dpi.
    fig.savefig(os.path.join(OUT, "results_chart.png"), dpi=150)
    # Close the figure to free memory.
    plt.close(fig)

    # Draw the length chart only if at least one word model has predictions.
    if length_curves:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        # One line per run: CER (in percent) for each length group.
        for name, curve in length_curves.items():
            ax.plot([b for b, _ in curve], [100 * c for _, c in curve], marker="o", label=name)
        ax.set_xlabel("word length (Unicode code points)")
        ax.set_ylabel("CER (%)")
        ax.set_title("CER by word length (test set)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        # Save the line chart.
        fig.savefig(os.path.join(OUT, "cer_by_length.png"), dpi=150)
        plt.close(fig)

    # Tell the user which files were written.
    print(f"wrote {OUT}/error_analysis.md, results_chart.png" + (", cer_by_length.png" if length_curves else ""))


# Run main() only when the file is started as a script, not when it is imported.
if __name__ == "__main__":
    main()
