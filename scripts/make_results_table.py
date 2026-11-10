"""Print a Markdown table of every results/*/metrics.json (run from the repository root).

Usage: python scripts/make_results_table.py
Paste the output into README.md and report Section 7. Missing runs are simply not listed.
"""
import glob
import json

rows = []
for path in sorted(glob.glob("results/*/metrics.json")):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    rows.append(m)

print("| Run | Model | Train subset | Decoding | CER | WER | Accuracy | Params | Train min | ms/image |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in rows:
    wer = m.get("wer")
    print("| {run} | {model} | {sub} | {dec} | {cer:.2f} % | {wer} | {acc:.2f} % | {p:,} | {t} | {ms} |".format(
        run=m["run_name"], model=m["model"], sub=m["train_subset"], dec=m["decoding"],
        cer=100 * m["cer"], wer="n/a" if wer is None else f"{100 * wer:.2f} %",
        acc=100 * m["accuracy"], p=m["params"], t=m["train_time_min"], ms=m["ms_per_image"]))
