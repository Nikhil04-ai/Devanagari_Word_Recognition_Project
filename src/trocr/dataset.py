"""Dataset and loaders for the TrOCR model (reads the Contract-1 CSV manifests, same files as the CRNN)."""
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .tokenizer import IGNORE_INDEX, CharTokenizer


def get_image_processor(pretrained):
    """Image preprocessing of the pre-trained TrOCR (resize to 384x384, normalise to [-1, 1]).
    Only the image processor is loaded, so the English text tokenizer is never needed."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(pretrained)


class TrOCRWordDataset(Dataset):
    """One item = {"pixel_values": FloatTensor(3,384,384), "labels": LongTensor(L), "text": str, "path": str}.

    `first_n` keeps the first N rows (overfit test); `random_n` keeps a random sample of N rows (seeded)."""

    def __init__(self, csv_path, tokenizer, image_processor, max_length=32, root=".",
                 first_n=0, random_n=0, seed=42):
        df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, keep_default_na=False)
        if first_n > 0:
            df = df.iloc[:first_n]
        elif 0 < random_n < len(df):
            df = df.sample(n=random_n, random_state=seed).sort_index()
        self.paths = df["image_path"].tolist()
        self.texts = df["text"].tolist()
        self.tok, self.proc, self.max_length, self.root = tokenizer, image_processor, max_length, root
        if self.paths and not os.path.exists(os.path.join(root, self.paths[0])):
            raise FileNotFoundError(
                f"first image not found: {os.path.join(root, self.paths[0])}. "
                "The CSV paths are relative to the repository root (for example data/raw/iiit_hw_dev/HindiSeg/...). "
                "Run from the repo root, extract the images to data/raw/iiit_hw_dev/, or pass --image_root.")

    def __len__(self):
        return len(self.paths)

    def load_image(self, i):
        return Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")

    def __getitem__(self, i):
        pixel_values = self.proc(images=self.load_image(i), return_tensors="pt").pixel_values[0]
        labels = torch.tensor(self.tok.encode(self.texts[i], self.max_length), dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels, "text": self.texts[i], "path": self.paths[i]}


def collate_fn(batch):
    """Stack images; pad labels with -100 so the loss ignores the padding."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    longest = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = b["labels"]
    return {"pixel_values": pixel_values, "labels": labels,
            "texts": [b["text"] for b in batch], "paths": [b["path"] for b in batch]}


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def get_loaders(cfg):
    """Build the tokenizer from vocab.txt (training split only) and the train / val DataLoaders.

    If cfg.overfit_n > 0, train AND val use only the first N training samples (sanity check: CER -> 0)."""
    d = cfg.data_dir
    tok = CharTokenizer.from_vocab_file(os.path.join(d, "vocab.txt"))
    proc = get_image_processor(cfg.pretrained)
    kw = dict(tokenizer=tok, image_processor=proc, max_length=cfg.max_length, root=cfg.image_root)
    train_csv = os.path.join(d, cfg.train_csv)
    if cfg.overfit_n > 0:
        train_ds = TrOCRWordDataset(train_csv, first_n=cfg.overfit_n, **kw)
        val_ds = TrOCRWordDataset(train_csv, first_n=cfg.overfit_n, **kw)
    else:
        train_ds = TrOCRWordDataset(train_csv, random_n=cfg.train_limit, seed=cfg.seed, **kw)
        val_ds = TrOCRWordDataset(os.path.join(d, cfg.val_csv), random_n=cfg.val_limit, seed=cfg.seed, **kw)
    return tok, {
        "train": make_loader(train_ds, cfg.batch_size, True, cfg.num_workers),
        "val": make_loader(val_ds, max(cfg.batch_size, 32), False, cfg.num_workers),
    }


if __name__ == "__main__":
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_dir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, {"data_dir": a.data_dir, "num_workers": 0, "batch_size": 4})
    tok, loaders = get_loaders(cfg)
    batch = next(iter(loaders["train"]))
    print("pixel_values", tuple(batch["pixel_values"].shape), "| labels", tuple(batch["labels"].shape))
    print("texts   :", batch["texts"])
    print("decoded :", [tok.decode(r) for r in batch["labels"]])
