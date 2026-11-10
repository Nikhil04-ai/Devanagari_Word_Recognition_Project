"""Data loading for the CRNN-CTC word recogniser (Contract 1 CSVs + Contract 2 vocab).

Images are turned into grayscale tensors 64 px high with white = +1.0 and ink = -1.0.
Labels are Unicode code points. Index 0 is the CTC blank, real characters start at 1.
"""
import csv
import os
import unicodedata

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_H = 64
MIN_W = 128  # narrow images are padded on the right so there are enough CTC time steps
MAX_W = 512  # very wide images are squeezed to this width
DOWNSAMPLE = 4  # the CNN divides the width by 4


class CharVocab:
    """Maps characters to indices. Index 0 is reserved for the CTC blank."""

    BLANK = 0

    def __init__(self, chars):
        self.chars = list(chars)
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8", newline="") as f:
            lines = [line.rstrip("\r\n") for line in f]
        return cls([c for c in lines if c != ""])

    def __len__(self):
        """Number of classes the network predicts, including the blank."""
        return len(self.chars) + 1

    def encode(self, text, strict=True):
        text = unicodedata.normalize("NFC", str(text).strip())
        ids = []
        for ch in text:
            if ch in self.char2idx:
                ids.append(self.char2idx[ch])
            elif strict:
                raise ValueError(f"character {ch!r} (U+{ord(ch):04X}) is not in the vocabulary")
        return ids

    def indices_to_text(self, ids):
        return "".join(self.chars[i - 1] for i in ids if i > 0)


def min_ctc_input_length(label_ids):
    """Fewest time steps CTC needs for this label: its length plus one blank between equal neighbours."""
    repeats = sum(1 for a, b in zip(label_ids, label_ids[1:]) if a == b)
    return len(label_ids) + repeats


def load_image(src, img_h=IMG_H, min_w=MIN_W, max_w=MAX_W, invert=False):
    """Open an image path or PIL image and return a float tensor of shape (1, img_h, W)."""
    img = Image.open(src) if isinstance(src, (str, os.PathLike)) else src
    img = img.convert("L")
    w, h = img.size
    new_w = min(max(1, round(w * img_h / max(h, 1))), max_w)
    img = img.resize((new_w, img_h), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()) / 255.0
    if invert:
        x = 1.0 - x
    x = (x - 0.5) / 0.5  # background (white) -> +1.0
    if new_w < min_w:
        x = torch.nn.functional.pad(x, (0, min_w - new_w), value=1.0)
    return x.unsqueeze(0)


class WordDataset(Dataset):
    """Reads a Contract 1 CSV (image_path,text). Paths are relative to `root` (the repo root)."""

    def __init__(self, csv_path, vocab, root=".", augment=None, invert=False):
        self.vocab = vocab
        self.root = root
        self.augment = augment
        self.invert = invert
        with open(csv_path, encoding="utf-8", newline="") as f:
            self.samples = [(r["image_path"], r["text"]) for r in csv.DictReader(f)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        img = Image.open(os.path.join(self.root, path))
        if self.augment is not None:
            img = self.augment(img)
        x = load_image(img, invert=self.invert)
        label = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return x, label, text, path


def collate_fn(batch):
    """Pad images on the right with white to the widest image of the batch."""
    images, labels, texts, paths = zip(*batch)
    widths = torch.tensor([im.shape[-1] for im in images], dtype=torch.long)
    max_w = int(widths.max())
    padded = torch.ones(len(images), 1, IMG_H, max_w)
    for k, im in enumerate(images):
        padded[k, :, :, : im.shape[-1]] = im
    return {
        "images": padded,
        "targets": torch.cat(labels),
        "target_lengths": torch.tensor([len(l) for l in labels], dtype=torch.long),
        "widths": widths,
        "texts": list(texts),
        "paths": list(paths),
    }


def make_loader(csv_path, vocab, batch_size=64, shuffle=False, root=".", num_workers=2,
                augment=None, invert=False):
    ds = WordDataset(csv_path, vocab, root=root, augment=augment, invert=invert)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
