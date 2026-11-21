"""Dataset and loaders for the TrOCR model (reads the Contract-1 CSV manifests, same files as the CRNN).

Where it fits in the pipeline: this file feeds the training loop. It reads a CSV manifest
(columns `image_path` and `text`), opens each word image, turns it into the tensor that the
pre-trained TrOCR encoder expects, and turns the word text into label ids with the tokenizer.
DataLoaders then group the items into batches.

Flow for one sample:  CSV row -> image file -> RGB PIL image -> image processor -> pixel_values (3, 384, 384)
                                 text -> CharTokenizer.encode -> labels (L,)
"""
# os: build and check file paths.
import os

# pandas: read the CSV manifest. torch: tensors. PIL: open image files.
import pandas as pd
import torch
from PIL import Image
# Dataset is the base class we extend; DataLoader makes batches from a Dataset.
from torch.utils.data import DataLoader, Dataset

# IGNORE_INDEX (-100) pads the labels; CharTokenizer encodes the text labels.
from .tokenizer import IGNORE_INDEX, CharTokenizer


def get_image_processor(pretrained):
    """Image preprocessing of the pre-trained TrOCR (resize to 384x384, normalise to [-1, 1]).
    Only the image processor is loaded, so the English text tokenizer is never needed.

    Input: Hugging Face model name (cfg.pretrained). Returns: an image processor object.
    Why use the original processor: the pre-trained encoder was trained with exactly this
    resizing and normalisation, so new images must be prepared the same way."""
    # Imported inside the function so that this module can be imported (and tested) without
    # loading the transformers library.
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(pretrained)


class TrOCRWordDataset(Dataset):
    """One item = {"pixel_values": FloatTensor(3,384,384), "labels": LongTensor(L), "text": str, "path": str}.

    `first_n` keeps the first N rows (overfit test); `random_n` keeps a random sample of N rows (seeded).

    Images are loaded lazily in __getitem__ (not all at once), so memory use stays small."""

    def __init__(self, csv_path, tokenizer, image_processor, max_length=32, root=".",
                 first_n=0, random_n=0, seed=42):
        """Inputs: csv_path = manifest file; tokenizer = CharTokenizer; image_processor = from
        get_image_processor; max_length = longest label (with <eos>); root = folder that the CSV
        image paths are relative to; first_n / random_n = optional subset size; seed = for random_n."""
        # dtype=str keeps every column as text (no number guessing); keep_default_na=False stops
        # pandas from turning words like "NA" or "null" into missing values.
        df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
        if first_n > 0:
            # Overfit test: always the same first N rows.
            df = df.iloc[:first_n]
        elif 0 < random_n < len(df):
            # Random subset. random_state=seed makes it the same every run; sort_index restores the original row order.
            df = df.sample(n=random_n, random_state=seed).sort_index()
        # Keep only the paths and texts as plain lists; the images themselves are read later, one by one.
        self.paths = df["image_path"].tolist()
        self.texts = df["text"].tolist()
        self.tok, self.proc, self.max_length, self.root = tokenizer, image_processor, max_length, root
        # Fail early with a clear message if the images are not where the CSV says (checks only the first one).
        if self.paths and not os.path.exists(os.path.join(root, self.paths[0])):
            raise FileNotFoundError(
                f"first image not found: {os.path.join(root, self.paths[0])}. "
                "The CSV paths are relative to the repository root (for example data/raw/iiit_hw_dev/HindiSeg/...). "
                "Run from the repo root, extract the images to data/raw/iiit_hw_dev/, or pass --image_root.")

    def __len__(self):
        """Number of samples (rows kept from the CSV)."""
        return len(self.paths)

    def load_image(self, i):
        """Open image number i as a 3-channel RGB PIL image.
        Why convert("RGB"): files may be grayscale (1 channel), but the encoder expects 3 channels."""
        return Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")

    def __getitem__(self, i):
        """Return sample i as a dict (see the class docstring).
        `pixel_values` has shape (3, 384, 384) for the pre-trained processor; `labels` has shape (L,),
        L <= max_length, and ends with <eos>. `text` and `path` are kept for evaluation and error analysis."""
        # The processor returns a batch of 1, shape (1, 3, H, W); [0] drops the batch axis -> (3, H, W).
        pixel_values = self.proc(images=self.load_image(i), return_tensors="pt").pixel_values[0]
        # Text -> ids (ends with <eos>, no <bos>) as a 1-D LongTensor of shape (L,).
        labels = torch.tensor(self.tok.encode(self.texts[i], self.max_length), dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels, "text": self.texts[i], "path": self.paths[i]}


def collate_fn(batch):
    """Stack images; pad labels with -100 so the loss ignores the padding.

    Input: list of B items from TrOCRWordDataset. Returns a dict with
        pixel_values (B, 3, 384, 384), labels (B, L_max) padded with -100, texts (list of B str), paths (list of B str).
    Why a custom collate: label lengths differ between words, so they cannot be stacked directly."""
    # All images have the same size after the processor, so torch.stack works: B x (3,H,W) -> (B,3,H,W).
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    # Length of the longest label in this batch.
    longest = max(len(b["labels"]) for b in batch)
    # (B, L_max) filled with -100, then each real label is copied into the start of its row.
    labels = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = b["labels"]
    return {"pixel_values": pixel_values, "labels": labels,
            "texts": [b["text"] for b in batch], "paths": [b["path"] for b in batch]}


def make_loader(ds, batch_size, shuffle, num_workers):
    """Wrap a dataset in a DataLoader that yields batches made by collate_fn.

    shuffle=True for training (new order every epoch), False for validation / test.
    pin_memory speeds up copying to the GPU, so it is only switched on when CUDA exists.
    persistent_workers keeps the worker processes alive between epochs (allowed only when num_workers > 0)."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def get_loaders(cfg):
    """Build the tokenizer from vocab.txt (training split only) and the train / val DataLoaders.

    If cfg.overfit_n > 0, train AND val use only the first N training samples (sanity check: CER -> 0).

    Input: config namespace. Returns: (tokenizer, {"train": DataLoader, "val": DataLoader})."""
    d = cfg.data_dir
    # Vocabulary file made from the training split only.
    tok = CharTokenizer.from_vocab_file(os.path.join(d, "vocab.txt"))
    proc = get_image_processor(cfg.pretrained)
    # Arguments that are the same for the train and the validation dataset.
    kw = dict(tokenizer=tok, image_processor=proc, max_length=cfg.max_length, root=cfg.image_root)
    train_csv = os.path.join(d, cfg.train_csv)
    if cfg.overfit_n > 0:
        # Overfit test: the same first N training samples are used for training and validation.
        train_ds = TrOCRWordDataset(train_csv, first_n=cfg.overfit_n, **kw)
        val_ds = TrOCRWordDataset(train_csv, first_n=cfg.overfit_n, **kw)
    else:
        # train_limit / val_limit = 0 means "use all rows".
        train_ds = TrOCRWordDataset(train_csv, random_n=cfg.train_limit, seed=cfg.seed, **kw)
        val_ds = TrOCRWordDataset(os.path.join(d, cfg.val_csv), random_n=cfg.val_limit, seed=cfg.seed, **kw)
    return tok, {
        # Training: shuffled. Validation: not shuffled, and the batch size is at least 32.
        "train": make_loader(train_ds, cfg.batch_size, True, cfg.num_workers),
        "val": make_loader(val_ds, max(cfg.batch_size, 32), False, cfg.num_workers),
    }


# Quick self-check: run `python -m src.trocr.dataset --data_dir data/processed/words`.
# It prints the shape of one training batch and checks that labels decode back to the text.
if __name__ == "__main__":
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args()
    # Small batch and no worker processes keep this check simple and fast.
    cfg = load_config(a.config, {"data_dir": a.data_dir, "num_workers": 0, "batch_size": 4})
    tok, loaders = get_loaders(cfg)
    batch = next(iter(loaders["train"]))
    print("pixel_values", tuple(batch["pixel_values"].shape), "| labels", tuple(batch["labels"].shape))
    print("texts   :", batch["texts"])
    print("decoded :", [tok.decode(r) for r in batch["labels"]])
