# Devanagari Word Recognition: CRNN-CTC vs Transformer OCR

Handwritten Hindi (Devanagari) recognition is a low-resource OCR problem. Words are joined by a headline (shirorekha), vowel signs (matras) attach on all sides of a letter, and half-letters form conjuncts. This project compares three model families on the same script, at character level and at word level:

| Model | Input → Output | Dataset |
|---|---|---|
| **A: CNN classifier** | 32×32 character image → one of 46 classes | [UCI Devanagari Handwritten Character Dataset (DHCD)](https://archive.ics.uci.edu/dataset/389) |
| **B: CRNN (CNN + BiLSTM + CTC)** | Word image → Unicode text | IIIT-HW-Dev (CVIT, IIIT Hyderabad) |
| **C: TrOCR-style encoder-decoder** | Word image → Unicode text | IIIT-HW-Dev |

All three are scored with the same metric code (`src/common/metrics.py`). The CNN is tested on isolated characters and the word models on whole words, so their error rates are **not directly comparable**; the report explains this.

## Results

Test sets, each evaluated once. Numbers come from `results/*/metrics.json`; `python scripts/make_results_table.py` prints the full table.

| Model | Test data | Level | Decoding | CER ↓ | WER ↓ | Accuracy ↑ | Parameters | ms / image |
|---|---|---|---|---|---|---|---|---|
| A: CNN | DHCD, 13,800 characters | Character | argmax | 0.30 % | n/a | 99.72 % | 1.2 M | 1.0 |
| B: CRNN-CTC | IIIT-HW-Dev, 12,869 words | Word | greedy | 6.59 % | 26.05 % | 73.95 % | 7.7 M | 3.6 |
| C: TrOCR-style | IIIT-HW-Dev, 12,869 words | Word | greedy | 6.47 % | 22.32 % | 77.68 % | 28.9 M | 70.6 |

The CNN reads isolated characters and the word models read whole words, so their error rates are not directly comparable. The test writers never appear in training, but every test word does, so the word results measure generalisation to new writers, not to new words. Error analysis: [results/summary/error_analysis.md](results/summary/error_analysis.md).

## What each member did

| Member | Work |
|---|---|
| **Nikhil** | Repository and shared code; IIIT-HW-Dev data preparation; CRNN-CTC model |
| **Prabhat** | TrOCR-style model |
| **Ravi** | CNN character classifier; results table; error analysis |
| **Monu** | README; literature review; report sections 1, 2, 8 and the merged report |

## Getting started

```bash
git clone https://github.com/Nikhil04-ai/Devanagari_Word_Recognition_Project.git
cd Devanagari_Word_Recognition_Project
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # all tests should pass
```

Run every command from the repository root. Datasets and checkpoints are not stored in Git.

## Data

**IIIT-HW-Dev (words).** Version 1 is a direct download (1.8 GB). The CVIT server limits each connection, so use the parallel downloader (resumable):

```bash
python scripts/download_iiit.py --out C:/dl_data/IIIT-HW-Hindi_v1.tar.gz
mkdir C:\dl_data\iiit_hw_dev
tar -xzf C:\dl_data\IIIT-HW-Hindi_v1.tar.gz -C C:\dl_data\iiit_hw_dev
tar -xzf C:\dl_data\iiit_hw_dev\HindiSeg.tar.gz -C C:\dl_data\iiit_hw_dev     # the images are in a second archive
mklink /J data\raw\iiit_hw_dev C:\dl_data\iiit_hw_dev      # keep the images outside OneDrive
```

`scripts/prepare_iiit.py` turns the raw files into `data/processed/words/{train,val,test}.csv`, `vocab.txt` and `stats.json` (the formats of [CONTRACTS.md](CONTRACTS.md)). **These files are already committed**, so you only need the images in `data/raw/iiit_hw_dev/`. Re-running the script takes about 9 minutes (it opens all 95,430 images).

**DHCD (characters).** Download the zip from the [UCI page](https://archive.ics.uci.edu/dataset/389) into `data/raw/dhcd/`; the loader reads the zip directly (an unzipped `Train/` and `Test/` also works).

## Train and evaluate

Every model follows the same pattern: an overfit check on 100 images, a full run that stops at its time budget, `--resume` after a disconnect, and one evaluation on the test set that writes `results/<model>_<subset>_<decoding>/`.

```bash
# Model A: CNN on DHCD (about 1.5 min on a T4 GPU)
python -m src.char_cnn.train --config configs/char_cnn/overfit.yaml     # sanity check: memorise 100 images
python -m src.char_cnn.train --config configs/char_cnn/full.yaml
python -m src.char_cnn.train --config configs/char_cnn/full.yaml --resume
python -m src.char_cnn.evaluate --ckpt checkpoints/char_cnn_full/best.pt   # -> results/char_cnn_full_argmax/

# Model B: CRNN-CTC (about 1.5 h on a T4 GPU)
python -m src.crnn.train --config configs/crnn/overfit.yaml
python -m src.crnn.train --config configs/crnn/full.yaml
python -m src.crnn.train --config configs/crnn/full.yaml --resume
python -m src.crnn.evaluate --ckpt checkpoints/crnn_full/best.pt          # -> results/crnn_full_greedy/

# Model C: TrOCR-style (downloads microsoft/trocr-small-handwritten on first use; about 1.5 h on a GPU)
python -m src.trocr.train --config configs/trocr/overfit.yaml
python -m src.trocr.train --config configs/trocr/full.yaml
python -m src.trocr.train --config configs/trocr/full.yaml --resume
python -m src.trocr.evaluate --ckpt checkpoints/trocr_full/best.pt        # -> results/trocr_full_greedy/
```

Training options can be overridden on the command line, for example `--epochs 3`, `--max_minutes 30` or `--overfit_n 20`; the word-model evaluations accept `--limit 200` for a quick check on the first test images. Ready-made Kaggle and Colab notebooks are in `notebooks/<member>/`.

## Results table, error analysis and report

```bash
python scripts/make_results_table.py           # Markdown table of every results/*/metrics.json
python results/summary/error_analysis.py       # confusions, error types, CER by word length
python report/tools/word_frequency.py          # report Section 7.4: test error by word frequency in training
python report/tools/build_report.py            # merge report/sections/ into report/build/report_draft.pdf
python report/tools/build_report.py --final    # report/P09_report.pdf (not added to Git)
```

The PDF is printed with Microsoft Edge (or Chrome), which shows Hindi text correctly. See [report/README.md](report/README.md).

There is no GPU requirement for the tests, but training needs one. See [docs/KAGGLE.md](docs/KAGGLE.md) for running on a free Kaggle GPU.

## Repository layout

```
src/common/   shared metrics (CER/WER/accuracy), results writer, seed
src/crnn/     Model B: data loader, model, greedy decoding, training, evaluation
src/char_cnn/ Model A (CNN character classifier)
src/trocr/    Model C (TrOCR-style)
configs/      YAML experiment configs
scripts/      data download and preparation, results table
results/      metrics and predictions of each run; summary/ has the chart and error analysis
report/       report sections, references, figures and the scripts that build the PDF
notebooks/    Kaggle and Colab runners for each model
docs/         Kaggle guide
tests/        unit tests
```

The agreed data formats and prediction API are in [CONTRACTS.md](CONTRACTS.md); how we work is in [CONTRIBUTING.md](CONTRIBUTING.md); the report is assembled from [report/](report/README.md).

## Reproducibility

- Seed 42 everywhere (`src/common/seed.py`).
- Labels are Unicode NFC; the character list is built from the training split only.
- The test split is evaluated once, after training; the validation split is used for every choice.
- Every run writes `results/<run_name>/test_predictions.csv` and `metrics.json` through `src/common/io.py`.

## References

1. Shi, Bai, Yao. *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition (CRNN)*, TPAMI 2017. https://arxiv.org/abs/1507.05717
2. Li et al. *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*, AAAI 2023. https://arxiv.org/abs/2109.10282
3. Dutta, Krishnan, Mathew, Jawahar. *Offline Handwriting Recognition on Devanagari using a new Benchmark Dataset*, DAS 2018 (IIIT-HW-Dev).
4. Acharya, Pant, Gyawali. *Deep Learning Based Large Scale Handwritten Devanagari Character Recognition*, SKIMA 2015 (DHCD).

The full list is in [report/references.md](report/references.md). DHCD is licensed CC BY 4.0.
