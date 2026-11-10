# Devanagari Word Recognition: CRNN-CTC vs Transformer OCR

Handwritten Hindi (Devanagari) recognition is a low-resource OCR problem. Words are joined by a headline (shirorekha), vowel signs (matras) attach on all sides of a letter, and half-letters form conjuncts. This project compares three model families on the same script, at character level and at word level.

## What we are building

| Model | Input → Output | Dataset |
|---|---|---|
| **A: CNN classifier** | 32×32 character image → one of 46 classes | [UCI Devanagari Handwritten Character Dataset (DHCD)](https://archive.ics.uci.edu/dataset/389) |
| **B: CRNN (CNN + BiLSTM + CTC)** | Word image → Unicode text | IIIT-HW-Dev (CVIT, IIIT Hyderabad) |
| **C: TrOCR-style encoder-decoder** | Word image → Unicode text | IIIT-HW-Dev |

## What we will do

1. **Prepare the data.** Clean the IIIT-HW-Dev word images, normalise all labels to Unicode NFC, build the character vocabulary, and create fixed train/validation/test splits plus 10%, 25% and 50% training subsets.
2. **Train the three models.** The CNN classifies single characters. The CRNN reads a word as a sequence and is trained with CTC loss. The TrOCR-style model fine-tunes a pre-trained Vision Transformer encoder with a Devanagari character decoder.
3. **Compare them.** Report character error rate (CER) and word error rate (WER) for every model, with greedy, beam-search and lexicon decoding for the CRNN.
4. **Test low-resource behaviour.** Train the CRNN and TrOCR on 10%, 25%, 50% and 100% of the data and plot CER against training-set size.
5. **Analyse the errors.** Find which characters, matras and conjuncts are confused, and how error depends on word length.
6. **Build a demo.** A Gradio app that takes an uploaded image and shows each model's prediction.

All models are evaluated with the same metric code so the numbers are directly comparable.

## Contributions

| Member | Work |
|---|---|
| **Nikhil** | Project setup and repository; toy and real word dataset pipeline; CRNN-CTC model |
| **Prabhat** | TrOCR model; data augmentation; exploratory data analysis |
| **Ravi** | CNN character classifier; results tables and plots; Gradio demo |
| **Monu** | Error analysis; README; project report and slides |

## Repository layout

Created step by step as the project progresses.

```
src/         model, data and shared code (common/, crnn/, trocr/, char_cnn/, data_words/)
configs/     YAML experiment configs
scripts/     dataset preparation scripts
notebooks/   data exploration and error analysis
app/         Gradio demo
results/     metrics, predictions and plots
report/      report sections
```

Datasets and trained checkpoints are not stored in Git. The agreed data formats and APIs are in [CONTRACTS.md](CONTRACTS.md).

## Getting started

```bash
git clone https://github.com/Nikhil04-ai/Devanagari_Word_Recognition_Project.git
cd Devanagari_Word_Recognition_Project
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q             # all tests should pass
```

Run every command from the repository root.

## References

1. Shi, Bai, Yao. *An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)*, TPAMI 2017. https://arxiv.org/abs/1507.05717
2. Li et al. *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*, AAAI 2023. https://arxiv.org/abs/2109.10282
