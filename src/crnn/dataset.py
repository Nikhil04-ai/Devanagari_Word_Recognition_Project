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

# ===== CONCEPTS USED IN THIS FILE =====
# CSV manifest: a table file (image_path,text) listing every image and its true word. WordDataset reads it.
# Vocabulary (vocab.txt): the list of all characters the model may output. CharVocab loads it from file.
# Code point: the number Unicode gives to one character. A Devanagari letter, matra or nukta is one code point;
#    labels are split into code points in CharVocab.encode.
# Label encoding: each character becomes an integer index. Real characters are 1..N, shifted by +1.
# CTC blank: index 0, a special "no character" class that never appears in a real label (CharVocab.BLANK).
# Unicode NFC: a standard way to write the same text, so identical-looking Devanagari has identical code points.
#    Used in CharVocab.encode.
# Repeated letters and CTC: two equal neighbours need a blank between them, so min_ctc_input_length adds 1 per pair.
# Time steps: image columns after the CNN (width / 4 = DOWNSAMPLE). More time steps than label length are needed.
# Grayscale and normalisation: image -> 1 channel, pixels (x - 0.5) / 0.5 give -1.0 (ink) .. +1.0 (white) in
#    load_image.
# Aspect ratio and resize: height fixed to IMG_H = 64, width scaled in proportion, limited to MIN_W..MAX_W.
# Padding: extra white columns added on the right. load_image pads to MIN_W; collate_fn pads to the widest image.
# Invert: flips light and dark for images that have light ink on a dark background (the invert option).
# Augmentation: random changes to a training image (optional augment function, applied on the PIL image).
# Dataset / DataLoader: PyTorch classes. Dataset gives one sample; DataLoader groups samples into batches.
# collate_fn: the function that turns a list of samples into one batch (needed because sizes differ).
# Concatenated targets: for nn.CTCLoss all labels are joined into one 1-D tensor plus a list of lengths.
# num_workers / pin_memory: background reader processes / faster CPU-to-GPU copy in make_loader.
# shuffle: random sample order each epoch for training; fixed order for validation and test.

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: The data side of the CRNN: it turns the CSV rows and word images into padded tensor batches and encoded labels.
# Q: How is it run?
# A: It is not run directly. train.py and evaluate.py import WordDataset, CharVocab, collate_fn and load_image.
# Q: What does it read and write?
# A: It reads the train/val/test CSVs, vocab.txt and the image files. It writes nothing to disk.
# Q: Why is there a separate vocabulary file and a blank at index 0?
# A: The model needs fixed class numbers. vocab.txt gives the same character order in training and testing, and
#    index 0 is the CTC blank (the PyTorch default), so real characters start at 1.

IMG_H = 64 # every image is resized to this height (a fixed height lets the CNN output a fixed 4 rows)
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
    # Q: What does this class do and why does it exist?
    # A: It converts characters to class numbers and back, and it knows the blank index. The network and the
    #    CTC loss work with numbers, so every label and every prediction passes through this class.
    # Q: Why is BLANK a class constant set to 0?
    # A: train.py and decode.py read CharVocab.BLANK, so the blank index is written in one place only. It is
    #    0 because that is the nn.CTCLoss default, which also pushes real characters to start at 1.

    BLANK = 0

    def __init__(self, chars):
        """chars: iterable of single characters (strings of length 1), in the order of vocab.txt."""
        # Q: What does __init__ do and why does it exist?
        # A: It stores the character list and builds a dictionary from character to index for fast look-up.
        # Q: Why i + 1 in the dictionary?
        # A: Index 0 belongs to the blank. Without the +1 the first character would get the blank's index and
        #    CTC would treat it as "no character".
        self.chars = list(chars)
        # char -> index, shifted by 1 because index 0 belongs to the blank.
        self.char2idx = {c: i + 1 for i, c in enumerate(self.chars)}

    @classmethod
    def from_file(cls, path):
        """Build a vocab from vocab.txt (one character per line, written by prepare_iiit.py)."""
        # Q: What does it do and why does it exist?
        # A: It reads vocab.txt and builds a CharVocab from it, so training and evaluation use the same
        #    characters in the same order, as written by prepare_iiit.py.
        # Q: Why newline="" and rstrip("\r\n") instead of a plain strip()?
        # A: strip() would also remove a real space or other whitespace character that is part of the vocabulary.
        #    Only the line break is removed, and empty lines are skipped.
        # newline="" and rstrip("\r\n"): remove only the line break, nothing else on the line.
        with open(path, encoding="utf-8", newline="") as f:
            lines = [line.rstrip("\r\n") for line in f]
        # Skip empty lines (for example the one after the last newline).
        return cls([c for c in lines if c != ""])

    def __len__(self):
        """Number of classes the network predicts, including the blank."""
        # Q: What does it do and why does it exist?
        # A: len(vocab) gives the number of output classes, which is used as num_classes when the CRNN is built.
        # Q: Why + 1?
        # A: The blank is an extra class that is not in self.chars. If it were forgotten, the final layer would
        #    be one output too small and the largest character index would be out of range.
        return len(self.chars) + 1

    def encode(self, text, strict=True):
        """Turn a string into a list of class indices (ints, each >= 1).

        strict=True: raise ValueError for a character that is not in the vocabulary.
        strict=False: silently skip such characters.
        """
        # Q: What does it do and why does it exist?
        # A: It turns a label word into the list of integers that nn.CTCLoss needs as the target. It first
        #    strips the text and normalises it to NFC.
        # Q: Why is strict=True the default?
        # A: A label with an unknown character would be trained on a wrong target without any warning. The error
        #    stops the run so the data problem is fixed. strict=False is only for when skipping is acceptable.
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
        # Q: What does it do and why does it exist?
        # A: It is the reverse of encode: indices become a string. decode.py uses it to turn predicted indices
        #    into text.
        # Q: Why i - 1 and why skip i = 0?
        # A: Indices were shifted by +1 in __init__, so i - 1 gives the position in self.chars. Index 0 is the
        #    blank, which has no character, so it is left out.
        # i - 1 undoes the +1 shift made in __init__.
        return "".join(self.chars[i - 1] for i in ids if i > 0)


def min_ctc_input_length(label_ids):
    """Fewest time steps CTC needs for this label: its length plus one blank between equal neighbours.

    label_ids: list of class indices. Returns an int.
    Why: when two equal characters are next to each other, CTC must put a blank between them.
    Without it, the decoder would merge them into one character.
    If the model has fewer time steps than this, the loss is infinite (the label cannot fit).
    """
    # Q: What does it do and why does it exist?
    # A: It returns the smallest number of time steps in which CTC can still write this label. It is a helper
    #    to check whether the CNN output is long enough for a word (the loss is infinite when it is not).
    # Q: Why add the number of repeats?
    # A: Two equal neighbouring characters need a blank between them, or decoding would merge them into one.
    #    So the label needs one extra time step for each such pair.
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
    # Q: What does it do and why does it exist?
    # A: It turns one image into the network's input tensor: grayscale, height 64, width in proportion, values
    #    -1..+1, with white padding if it is narrow. Training, evaluation and timing all use it.
    # Q: Why keep the aspect ratio and pad narrow images to min_w?
    # A: Squashing the width would distort the letters. The CNN divides the width by 4, and a short word needs
    #    enough time steps for CTC, so narrow images are padded on the right with white (+1.0).
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
    # Q: What does this class do and why does it exist?
    # A: It is a PyTorch Dataset: it knows how many words there are and returns one (image, label) pair
    #    for an index. A DataLoader uses it to build batches.
    # Q: Why does it keep only paths and open images in __getitem__?
    # A: Keeping every image in memory would not fit for large data. Reading on demand (with worker processes)
    #    uses little memory.

    def __init__(self, csv_path, vocab, root=".", augment=None, invert=False):
        """csv_path: CSV with columns image_path,text. vocab: CharVocab.
        root: folder that image_path is relative to. augment: optional function PIL image -> PIL image.
        invert: passed on to load_image.
        """
        # Q: What does __init__ do and why does it exist?
        # A: It stores the settings and reads the CSV into a list of (image_path, text) pairs.
        # Q: Why csv.DictReader and utf-8 with newline=""?
        # A: DictReader picks columns by name (image_path, text), so column order does not matter. utf-8 is
        #    needed for Devanagari text, and newline="" is what the csv module requires for correct line breaks.
        self.vocab = vocab
        self.root = root
        self.augment = augment
        self.invert = invert
        # Only the paths and texts are stored here. The images are opened later in __getitem__.
        with open(csv_path, encoding="utf-8", newline="") as f:
            self.samples = [(r["image_path"], r["text"]) for r in csv.DictReader(f)]

    def __len__(self):
        """Number of samples (rows in the CSV)."""
        # Q: What does it do and why does it exist?
        # A: It tells the DataLoader how many samples exist, so it knows how many batches make one epoch.
        # Q: Why use len(self.samples) and not the number of files on disk?
        # A: train.py and evaluate.py may cut self.samples (overfit_n, val_limit, --limit), so the list is the
        #    single source of truth for the size.
        return len(self.samples)

    def __getitem__(self, i):
        """Return (image (1, 64, W) float tensor, label (L,) long tensor, text str, path str)."""
        # Q: What does it do and why does it exist?
        # A: It loads sample number i: opens the image, optionally augments it, converts it to a tensor and
        #    encodes the text into a label tensor. The DataLoader calls it for each index of a batch.
        # Q: Why is augmentation applied before load_image?
        # A: The augment function works on a PIL image, and load_image then does the tensor conversion, resize
        #    and normalisation. Doing it in the other order would not fit the function's input type.
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
    # Q: What does it do and why does it exist?
    # A: It joins a list of samples into one batch dict: images padded to the same width, labels concatenated
    #    into one long tensor, plus lengths and widths. The DataLoader calls it for every batch.
    # Q: Why keep widths and target_lengths?
    # A: Padding hides the real size. widths give the valid time steps per image, and target_lengths let CTC
    #    cut the long targets tensor back into single labels. Wrong lengths would make CTC read padding.
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
    # Q: What does it do and why does it exist?
    # A: It builds a WordDataset from a CSV and wraps it in a DataLoader with collate_fn. It is a short way
    #    to get ready-to-use batches from one CSV file.
    # Q: Why shuffle=True only for training?
    # A: A new random order each epoch helps the model learn. Validation and test keep a fixed order so
    #    results can be compared and repeated.
    ds =WordDataset(csv_path, vocab, root=root, augment=augment, invert=invert)
    # pin_memory only helps (and only works) when a GPU is used: it speeds up the copy to the GPU.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
