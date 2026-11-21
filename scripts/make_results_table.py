"""Print a Markdown table of every results/*/metrics.json (run from the repository root).

Usage: python scripts/make_results_table.py
Paste the output into README.md and report Section 7. Missing runs are simply not listed.

Where this file fits: the last step. Every model writes metrics.json through src/common/io.py
in the same format, so one table can compare them all (CER, WER, accuracy, size, time, speed).
"""
# glob finds the metrics files; json reads them.
import glob
import json

rows = []
# sorted() gives a stable row order (alphabetical by run folder name).
for path in sorted(glob.glob("results/*/metrics.json")):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    rows.append(m)

# Table header and the separator line that Markdown needs.
print("| Run | Model | Train subset | Decoding | CER | WER | Accuracy | Params | Train min | ms/image |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in rows:
    wer = m.get("wer")  # some models may not report WER, so use .get() (gives None if missing)
    # CER, WER and accuracy are stored as fractions, so *100 shows them as percent.
    # params uses {p:,} for thousands separators.
    print("| {run} | {model} | {sub} | {dec} | {cer:.2f} % | {wer} | {acc:.2f} % | {p:,} | {t} | {ms} |".format(
        run=m["run_name"], model=m["model"], sub=m["train_subset"], dec=m["decoding"],
        cer=100 * m["cer"], wer="n/a" if wer is None else f"{100 * wer:.2f} %",
        acc=100 * m["accuracy"], p=m["params"], t=m["train_time_min"], ms=m["ms_per_image"]))
