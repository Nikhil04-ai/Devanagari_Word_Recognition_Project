"""Data loading for the CRNN-CTC word recogniser (Contract 1 CSVs + Contract 2 vocab).

Images are turned into grayscale tensors 64 px high with white = +1.0 and ink = -1.0.
Labels are Unicode code points. Index 0 is the CTC blank, real characters start at 1.

Where this file fits in the pipeline:
    scripts/prepare_iiit.py writes train/val/test CSVs and vocab.txt
        -> dataset.py (this file) reads them and builds batches
        -> model.py / train.py / evaluate.py consume the batches.

What a batch looks like (see collate_fn): padded images, all labels joined into one long
1-D tensor, the length of each label, the real width of each image, and the raw texts and paths.
"""
# Standard library: read CSV files, build file paths, Unicode normalisation.
import csv
import os
import unicodedata

# numpy / torch / PIL: image to array to tensor. Dataset and DataLoader: PyTorch batching tools.
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_H = 64  # every image is resized to this height (a fixed height lets the CNN output a fixed 4 rows)
MIN_W = 128  # narrow images are padded on the right so there are enough CTC time steps
MAX_W = 512  # very wide images are squeezed to this width
DOWNSAMPLE = 4  # the CNN divides the width by 4
# Note: scripts/prepare_iiit.py repeats IMG_H, MIN_W, MAX_W and DOWN. The values must stay the same.


class CharVocab:
    """Maps characters to indices. Index 0 is reserved for the CTC blank.

    The blank must be a class of its own that never appears in a real label. CTC uses it to
    separate repeated letters (for example "ll") and to say "no character at this time step".
    Index 0 is the PyTorch default for nn.CTCLoss(blank=...), so real characters start at 1.
    """

    BLANK = 0

    def __init__(self, chars):
        """chars: iterable of single characters (strings of length 1), in the order of vocab.txt."""
        self.chars = list(chars)
        # char -> index, shifted by 1 because index 0 belongs to the blank.
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}

    @classmethod
    def from_file(cls, path):
        """Build a vocab from vocab.txt (one character per line, written by prepare_iiit.py)."""
        # newline="" and rstrip("\r\n"): remove only the line break, nothing else on the line.
        with open(path, encoding="utf-8", newline="") as f:
            lines = [line.rstrip("\r\n") for line in f]
        # Skip empty lines (for example the one after the last newline).
        return cls([c for c in lines if c != ""])

    def __len__(self):
        """Number of classes the network predicts, including the blank."""
        return len(self.chars) + 1

    def encode(self, text, strict=True):
        """Turn a string into a list of class indices (ints, each >= 1).

        strict=True: raise ValueError for a character that is not in the vocabulary.
        strict=False: silently skip such characters.
        """
        # NFC makes the same Devanagari word always use the same code points (matra / nukta order),
        # so a label and the vocab agree. The same normalisation is done in metrics.py.
        text = unicodedata.normalize("NFC", str(text).strip())
        ids = []
        for ch in text:
            if ch in self.char2idx:
                ids.append(self.char2idx[ch])
            elif strict:
                raise ValueError(f"character {ch!r} (U+{ord(ch):04X}) is not in the vocabulary")
        return ids

    def indices_to_text(self, ids):
        """Turn a list of class indices back into a string. Index 0 (blank) is skipped."""
        # i - 1 undoes the +1 shift made in __init__.
        return "".join(self.chars[i - 1] for i in ids if i > 0)


def min_ctc_input_length(label_ids):
    """Fewest time steps CTC needs for this label: its length plus one blank between equal neighbours.

    label_ids: list of class indices. Returns an int.
    Why: when two equal characters are next to each other, CTC must put a blank between them.
    Without it, the decoder would merge them into one character.
    If the model has fewer time steps than this, the loss is infinite (the label cannot fit).
    """
    # Count the places where a character is the same as the one before it.
    repeats = sum(1 for a, b in zip(label_ids, label_ids[1:]) if a == b)
    return len(label_ids) + repeats


def load_image(src, img_h=IMG_H, min_w=MIN_W, max_w=MAX_W, invert=False):
    """Open an image path or PIL image and return a float tensor of shape (1, img_h, W).

    src: file path or an already opened PIL image.
    invert: flip dark and light (for images with light ink on a dark background).
    Steps: grayscale -> resize to height img_h (keep aspect ratio, width capped at max_w)
    -> scale pixels to [-1, +1] -> pad on the right with white if narrower than min_w.
    """
    img = Image.open(src) if isinstance(src, (str, os.PathLike)) else src
    img = img.convert("L")  # "L" = 8-bit grayscale, so the network gets 1 channel
    w, h = img.size
    # Keep the aspect ratio: new width = w * img_h / h. At least 1 px, at most max_w.
    new_w = min(max(1, round(w * img_h / max(h, 1))), max_w)
    img = img.resize((new_w, img_h), Image.Resampling.BILINEAR)
    # PIL image -> float array (img_h, new_w) -> tensor in [0, 1]. .copy() makes the array writable,
    # which torch.from_numpy needs.
    x = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()) / 255.0
    if invert:
        x = 1.0 - x
    x = (x - 0.5) / 0.5  # background (white) -> +1.0
    if new_w < min_w:
        # Pad the right side with +1.0 (white = paper). pad = (left, right) for the last axis.
        x = torch.nn.functional.pad(x, (0, min_w - new_w), value=1.0)
    return x.unsqueeze(0)  # add the channel axis: (img_h, W) -> (1, img_h, W)


class WordDataset(Dataset):
    """Reads a Contract 1 CSV (image_path,text). Paths are relative to `root` (the repo root).

    Each item is one word image with its label. Images are read from disk on demand
    (not all kept in memory), so large datasets fit.
    """

    def __init__(self, csv_path, vocab, root=".", augment=None, invert=False):
        """csv_path: CSV with columns image_path,text. vocab: CharVocab.
        root: folder that image_path is relative to. augment: optional function PIL image -> PIL image.
        invert: passed on to load_image.
        """
        self.vocab = vocab
        self.root = root
        self.augment = augment
        self.invert = invert
        # Only the paths and texts are stored here. The images are opened later in __getitem__.
        with open(csv_path, encoding="utf-8", newline="") as f:
            self.samples = [(r["image_path"], r["text"]) for r in csv.DictReader(f)]

    def __len__(self):
        """Number of samples (rows in the CSV)."""
        return len(self.samples)

    def __getitem__(self, i):
        """Return (image (1, 64, W) float tensor, label (L,) long tensor, text str, path str)."""
        path, text = self.samples[i]
        img = Image.open(os.path.join(self.root, path))
        if self.augment is not None:
            img = self.augment(img)  # augmentation works on the PIL image, before the tensor conversion
        x = load_image(img, invert=self.invert)
        # strict encode: a label with an unknown character stops the run instead of training on a wrong label.
        label = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return x, label, text, path


def collate_fn(batch):
    """Pad images on the right with white to the widest image of the batch.

    batch: list of tuples from WordDataset.__getitem__.
    Returns a dict:
        images         (B, 1, 64, max_w) float tensor
        targets        (sum of label lengths,) all labels joined end to end (the format nn.CTCLoss accepts)
        target_lengths (B,) length of each label, so CTC can cut `targets` apart again
        widths         (B,) real width of each image before the batch padding
        texts, paths   lists of strings, kept for printing and for the results files
    Why a custom function: images have different widths and labels have different lengths,
    so PyTorch cannot stack them by default.
    """
    images, labels, texts, paths = zip(*batch)  # unzip the list of tuples into 4 tuples
    widths = torch.tensor([im.shape[-1] for im in images], dtype=torch.long)
    max_w = int(widths.max())
    # Start with all-white (1.0) and copy each image into the left part. The rest stays white padding.
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
    """Build a DataLoader for one CSV file.

    shuffle: True for training (new order each epoch), False for validation and test.
    num_workers: background processes that read and prepare images.
    """
    ds = WordDataset(csv_path, vocab, root=root, augment=augment, invert=invert)
    # pin_memory only helps (and only works) when a GPU is used: it speeds up the copy to the GPU.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
