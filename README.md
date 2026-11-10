# Devanagari Word Recognition: CRNN-CTC vs Transformer OCR

Deep Learning group project **P09**. The project recognises handwritten Hindi (Devanagari) characters and words, which is a low-resource OCR problem. It compares three model families:

| Model | Task | Dataset |
|---|---|---|
| **A: CNN classifier** | Single character → 1 of 46 classes | UCI Devanagari Handwritten Character Dataset (DHCD) |
| **B: CRNN (CNN + BiLSTM + CTC)** | Word image → Unicode text | IIIT-HW-Dev (CVIT, IIIT Hyderabad) |
| **C: TrOCR-style encoder-decoder** | Word image → Unicode text | IIIT-HW-Dev |

The main outputs are character and word error rates (CER and WER) across the three models, and a low-resource experiment that trains on 10%, 25%, 50% and 100% of the data.

> **Status:** repository initialised. Project setup (shared code, contracts, folder skeleton) is next.

## Team

| Member | Share | Responsibilities |
|---|---|---|
| Nikhil | 30% + setup | Project setup and team lead; toy and real word data pipeline; CRNN-CTC model |
| Prabhat | 30% | TrOCR model; augmentation; exploratory data analysis |
| Ravi | 20% | Character CNN; results summary; Gradio demo |
| Monu | 20% | Error analysis; README; report writing and merge |

The full plan, including the step order, file ownership and AI prompts for each member, is in [docs/P09_Devanagari_OCR_Team_Plan.pdf](docs/P09_Devanagari_OCR_Team_Plan.pdf).

## How we work

- Each member works on **their own branch**: `nikhil-crnn`, `prabhat-trocr`, `ravi-charcnn`, `monu-analysis`.
- Each member edits **only the files they own** (see Section 5 of the plan). Never run `git add .`.
- Open a pull request to `main` at least once a week. Nikhil reviews the others' PRs, and Prabhat reviews Nikhil's.
- Datasets and checkpoints are **never** committed. They are shared through a private team Drive or Kaggle dataset.

```bash
git clone https://github.com/Nikhil04-ai/Devanagari_Word_Recognition_Project.git
cd Devanagari_Word_Recognition_Project
git checkout -b <your-branch>
```

## References

1. Shi, Bai, Yao. *An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)*, TPAMI 2017. https://arxiv.org/abs/1507.05717
2. Li et al. *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*, AAAI 2023. https://arxiv.org/abs/2109.10282
