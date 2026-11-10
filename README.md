# Devanagari Word Recognition: CRNN-CTC vs Transformer OCR

Handwritten Hindi (Devanagari) recognition is a low-resource OCR problem. Words are joined by a headline (shirorekha), vowel signs (matras) attach on all sides of a letter, and half-letters form conjuncts. This project compares three model families on the same script, at character level and at word level:

| Model | Input → Output | Dataset |
|---|---|---|
| **A: CNN classifier** | 32×32 character image → one of 46 classes | [UCI Devanagari Handwritten Character Dataset (DHCD)](https://archive.ics.uci.edu/dataset/389) |
| **B: CRNN (CNN + BiLSTM + CTC)** | Word image → Unicode text | IIIT-HW-Dev (CVIT, IIIT Hyderabad) |
| **C: TrOCR-style encoder-decoder** | Word image → Unicode text | IIIT-HW-Dev |

All three are scored with the same metric code (`src/common/metrics.py`). The CNN is tested on isolated characters and the word models on whole words, so their error rates are **not directly comparable**; the report explains this.

## Results

Filled in when the runs finish (test set, used once). Numbers come from `results/*/metrics.json`.

| Model | Level | Decoding | CER ↓ | WER ↓ | Accuracy ↑ |
|---|---|---|---|---|---|
| A: CNN | Character | argmax | TODO | n/a | TODO |
| B: CRNN-CTC | Word | greedy | 6.59 % | 26.05 % | 73.95 % |
| C: TrOCR-style | Word | greedy | TODO | TODO | TODO |

## What each member did

| Member | Work |
|---|---|
| **Nikhil** | Repository and shared code; IIIT-HW-Dev data preparation; CRNN-CTC model |
| **Prabhat** | TrOCR-style model |
| **Ravi** | CNN character classifier; results table; error analysis |
| **Monu** | README; project report |

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

**DHCD (characters).** Download from the [UCI page](https://archive.ics.uci.edu/dataset/389) into `data/raw/dhcd/` (folders `Train/` and `Test/`).

## Train and evaluate

```bash
# CRNN-CTC (Model B)
python -m src.crnn.train --config configs/crnn/overfit.yaml     # sanity check: memorise 100 images
python -m src.crnn.train --config configs/crnn/full.yaml        # full run, stops at its time budget
python -m src.crnn.train --config configs/crnn/full.yaml --resume   # continue after a disconnect
python -m src.crnn.evaluate --ckpt checkpoints/crnn_full/best.pt    # test set, written to results/

# Models A and C follow the same pattern (in progress):
#   python -m src.char_cnn.train --config configs/char_cnn/cnn.yaml
#   python -m src.trocr.train    --config configs/trocr/full.yaml
```

There is no GPU requirement for the tests, but training needs one. See [docs/KAGGLE.md](docs/KAGGLE.md) for running on a free Kaggle GPU.

## Repository layout

```
src/common/   shared metrics (CER/WER/accuracy), results writer, seed
src/crnn/     Model B: data loader, model, greedy decoding, training, evaluation
src/char_cnn/ Model A (CNN character classifier)
src/trocr/    Model C (TrOCR-style)
configs/      YAML experiment configs
scripts/      data download and preparation
results/      metrics and predictions of each run
report/       report sections
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

The full list is in [report/references.md](report/references.md). DHCD is licensed CC BY 4.0.
