# Report

The submission is **one PDF, at most 10 MB, about 6 to 8 pages**. It is assembled from the files in `sections/`, in this order. Each owner writes their file; Monu merges them.

| # | File | Owner | Content |
|---|---|---|---|
| 1 | `sections/01_intro.md` | Monu | Problem, why Devanagari is hard, low-resource motivation, contributions |
| 2 | `sections/02_related_work.md` | Monu | CRNN, TrOCR, CTC, Indic datasets |
| 3 | `sections/03_data.md` | Nikhil | Datasets, preprocessing, splits, statistics |
| 4 | `sections/04_char_cnn.md` | Ravi | Model A: CNN character classifier |
| 5 | `sections/05_crnn.md` | Nikhil | Model B: CRNN-CTC |
| 6 | `sections/06_trocr.md` | Prabhat | Model C: TrOCR-style |
| 7 | `sections/07_results_analysis.md` | Monu (tables from Ravi) | Comparison table, error analysis |
| 8 | `sections/08_conclusion.md` | Monu | Findings, limitations, future work |
| 9 | `sections/09_contributions.md` | Monu (everyone confirms) | Who did what |
| - | `references.md` | Monu | Numbered reference list |

The abstract is at the top of `sections/01_intro.md`.

## Rules

- **Never invent a number.** Every number comes from `results/*/metrics.json` or from the data scripts. Write `TODO` until it exists, and search the files for `TODO` before merging.
- Replace the placeholders marked **TODO** and delete the helper notes in *italics*.
- Say in the report that the CNN's error (isolated characters) and the word models' error (whole words) are **not directly comparable**.
- Include a short paragraph on why this is a **low-resource** problem.
- State that every test word also occurs in training and that the test writers are new, so the test measures generalisation to new writers, not to new words.
- Cite the references by number: [1] CRNN, [2] TrOCR.

## Figures and file size

- Save figures as PNG, about 1600 px wide at most, each under about 200 KB. Put them in `report/figures/` named `<section>_<what>.png`, for example `05_crnn_loss_curve.png`.
- Use the Devanagari-capable font "Nirmala UI" (Windows) in plots so Hindi text is not shown as boxes.
- Check the final PDF size before submitting; the form rejects files over 10 MB.

## Merging into a PDF

From the repository root (needs Microsoft Edge or Chrome, and `pip install -r requirements.txt`):

```bash
python report/tools/build_report.py            # draft: report/build/report_draft.pdf (not in Git)
python report/tools/build_report.py --final    # report/Devanagari_Word_Recognition_Report.pdf
```

The script joins the sections in order, drops the `*Owner: ...*` lines, adds the numbered reference list from `references.md` (without its Status column) and prints the PDF with Edge, which shows Hindi correctly. It reports the page count and size, and lists every line that still has a TODO, `(check)` or an italic note. `--final` refuses to run while any is left. Then:

1. Open the PDF: check that every figure is mentioned in the text and Hindi text shows correctly.
2. Keep it to about 6 to 8 pages and under 10 MB.
3. The final file `report/Devanagari_Word_Recognition_Report.pdf` is the one file allowed in Git.

Section 7.4 comes from `python report/tools/word_frequency.py`, which reads `results/*/test_predictions.csv`.
