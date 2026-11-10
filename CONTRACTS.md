# Contracts (frozen interfaces)

These five agreements let everyone work in parallel without waiting for each other's code.
They are **frozen**: changing one needs all four members to agree, and Nikhil makes the change in a single PR.

## Contract 1: word dataset manifests (CSV)

```
data/processed/<dataset>/train.csv
data/processed/<dataset>/val.csv
data/processed/<dataset>/test.csv
data/processed/<dataset>/train_p10.csv   # nested subsets, seed 42: p10 ⊂ p25 ⊂ p50 ⊂ train
data/processed/<dataset>/train_p25.csv
data/processed/<dataset>/train_p50.csv

<dataset> = "toy" (synthetic, Day 2) or "words" (real IIIT-HW-Dev, week 2), both made by Nikhil

Columns (header row required, UTF-8, no BOM):
image_path,text
data/raw/iiit_hw_dev/images/0001/12.jpg,किताब

image_path : relative to the repository root
text       : NFC-normalised, no leading/trailing spaces
```

## Contract 2: vocabulary file

```
data/processed/<dataset>/vocab.txt
- one Unicode code point per line, sorted, UTF-8
- built from train.csv ONLY
- contains real characters only. Each model adds its own special tokens:
    CRNN  : index 0 = <blank>, characters start at 1
    TrOCR : 0 = <pad>, 1 = <bos>, 2 = <eos>, 3 = <unk>, characters start at 4
```

## Contract 3: shared metrics (`src/common/metrics.py`)

```python
cer(refs: list[str], hyps: list[str]) -> float
wer(refs: list[str], hyps: list[str]) -> float
accuracy(refs: list[str], hyps: list[str]) -> float
```

Never write your own metric. All strings are NFC-normalised inside these functions.

## Contract 4: results output (`src/common/io.py`)

```
save_results(run_name, image_paths, refs, hyps, extra) writes:
  results/<run_name>/test_predictions.csv   # image_path,ground_truth,prediction
  results/<run_name>/metrics.json

Required keys in `extra`:
  "model"          : "char_cnn" | "char_resnet18" | "crnn" | "trocr"
  "train_subset"   : "p10" | "p25" | "p50" | "full"
  "decoding"       : "argmax" | "greedy" | "beam" | "beam_lexicon"
  "params"         : number of trainable parameters (int)
  "train_time_min" : total training minutes (float)
  "ms_per_image"   : average inference time per image on GPU (float)

run_name convention: <model>_<subset>_<decoding>   e.g. crnn_p25_beam, trocr_full_beam, char_cnn_full_argmax
```

## Contract 5: prediction API (one per model)

```python
# src/char_cnn/predict.py (Ravi), src/crnn/predict.py (Nikhil), src/trocr/predict.py (Prabhat)
def load_model(ckpt_path: str, device: str = "cpu"):
    """Load the trained model and anything it needs (vocab, processor)."""

def predict(model, image: "PIL.Image.Image") -> str:
    """Return the recognised text (NFC). Accepts any size, RGB or grayscale."""
```

Until a model is ready, its `predict.py` is a stub returning `"(model not ready yet)"`, so the demo works from week 1.

## Shared conventions

- Python 3.10+, PyTorch 2.x. Seed 42 via `src/common/seed.py`.
- Configs are YAML files in `configs/<model>/`. Run with `python -m src.<model>.train --config configs/<model>/<name>.yaml`.
- Checkpoints go in `checkpoints/` (never in Git). Final checkpoints are shared on the team Drive.
- Save a checkpoint every epoch and support `--resume`, because Kaggle and Colab sessions disconnect.
- Run every command from the repository root.
