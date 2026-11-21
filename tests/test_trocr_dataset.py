"""Dataset / collate tests with a dummy image processor (no downloads).

What is tested: TrOCRWordDataset (CSV -> image tensor + label ids), collate_fn (batching with
-100 label padding), the first_n / random_n subset options, and the error for missing images.
Why a dummy processor: the real one is downloaded from Hugging Face; a fake one keeps the tests
fast and offline. Run with: pytest tests/test_trocr_dataset.py
"""
# types: SimpleNamespace builds a fake processor output.
import types

# pandas writes the test CSV, torch makes tensors, PIL makes small test images.
import pandas as pd
import torch
from PIL import Image

# The code under test.
from src.trocr.dataset import TrOCRWordDataset, collate_fn
from src.trocr.tokenizer import EOS, CharTokenizer

# ======================================================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================================================
# unit test: a small function that runs one piece of code on tiny input and checks the result. Every function whose
#     name starts with test_ in this file is one unit test; pytest finds and runs them.
# assert: a line that must be true. If it is false, the test fails. Used in every test here.
# fixture: a helper object that pytest creates and passes to a test by argument name. Here only the built-in
#     fixture tmp_path is used (see next item).
# tmp_path: a pytest fixture that gives a new empty temporary folder (a pathlib.Path) for each test, deleted
#     later. Our tests write fake images and a CSV there, so the real data folder is never touched.
# fake / synthetic data: data made by the test itself. make_csv() draws 30x10 white images with PIL and writes
#     a CSV with the texts "क", "कख", "कखग". Real IIIT-HW-Dev images are not needed.
# dummy (fake) object: DummyProc replaces the real image processor. It has the same call signature but returns
#     zeros, so the test is fast and needs no download from Hugging Face.
# CSV manifest: a table file with columns image_path and text; one row = one word image and its label.
#     TrOCRWordDataset reads it with pandas.
# Dataset: a PyTorch class with __len__ and __getitem__. ds[i] returns one sample (image tensor + label ids).
# image processor / pixel_values: turns a PIL image into the tensor the encoder expects. The real one gives
#     (3, 384, 384); DummyProc gives (3, 8, 8) after the dataset removes the batch axis with [0].
# RGB conversion: load_image calls .convert("RGB") so a grayscale ("L", 1 channel) file becomes 3 channels.
# tokenizer / label ids: CharTokenizer turns each character of the word into an integer id and appends <eos>.
# <eos>: "end of sequence" special token (id 2). The model learns to predict it when the word is finished.
# label padding with -100: words have different lengths, so collate_fn fills the short label rows with -100.
#     PyTorch cross-entropy ignores -100, so padding adds no loss.
# collate_fn / batch: the function a DataLoader uses to join a list of samples into one batch. It stacks images
#     with torch.stack and pads the labels to the longest one.
# first_n subset: keep only the first N rows of the CSV (used for the overfit sanity test).
# random_n subset + seed: keep N random rows. The seed fixes the random choice, so the same seed gives the same rows.
# lazy loading: images are opened in __getitem__, one by one, not all at once when the dataset is created.
# pytest.raises: a context manager that passes only if the code inside raises the given error. The match= text
#     is searched in the error message.
# relative path / root: CSV paths are relative; root is the folder they are joined to (here tmp_path).

# ======================================================================================================
# Q&A ABOUT THIS FILE
# ======================================================================================================
# Q: What part of the project does this file protect? A: The data side of the TrOCR model: src/trocr/dataset.py.
#    It checks that a CSV row becomes the right image tensor and label ids, that batching pads labels with -100,
#    that the subset options work, and that a wrong image folder gives a clear error.
# Q: How is it run? A: From the repository root with: pytest tests/test_trocr_dataset.py (or just pytest).
# Q: Why is no real dataset needed? A: make_csv() creates a few tiny fake images and a CSV in a temporary folder,
#    and DummyProc stands in for the downloaded image processor. The logic being tested (reading, tokenizing,
#    padding, subsetting) does not depend on what the pictures look like.


class DummyProc:
    """Fake image processor: ignores the image and always returns a zero tensor of shape (1, 3, 8, 8).
    It has the same call signature as the real processor, so the dataset code runs unchanged."""

    def __call__(self, images, return_tensors="pt"):
        # Q: What is this method for? A: It makes DummyProc callable like the real image processor:
        #    proc(images=..., return_tensors="pt").pixel_values. It is a helper, not a test.
        # Q: Why return zeros of shape (1, 3, 8, 8)? A: The dataset only needs a tensor with a leading batch axis
        #    of 1; it removes it with [0]. A small zero tensor is cheap, and the shape lets the tests check batching.
        # Q: Why does it ignore the image? A: These tests check the dataset logic, not the real resizing or
        #    normalisation. The real processor is downloaded and slow, so it is replaced.
        return types.SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8))


def make_csv(tmp_path, n=10):
    """Create n small image files and a CSV manifest in the pytest temp folder. Returns the CSV path.
    Row i has the text "क", "कख" or "कखग" (length 1, 2, 3, repeating), so label lengths differ."""
    # Q: What does this helper do? A: It writes n small PNG files and a CSV (columns image_path, text) into the
    #    temporary folder and returns the CSV path. Row i has the text "कखग"[: 1 + i % 3], so lengths are
    #    1, 2, 3, 1, ...
    # Q: Why are the images grayscale and the label lengths different on purpose? A: Grayscale ("L") checks that
    #    load_image converts to RGB. Different lengths force collate_fn to pad, so the -100 padding is really tested.
    # Q: Why is the image path stored as p.name only? A: The dataset joins root + path. The tests pass
    #    root=str(tmp_path), so the relative name is enough and the CSV does not hold machine-specific paths.
    rows = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.new("L", (30, 10), 255).save(p)  # grayscale on purpose: loader must convert to RGB
        rows.append((p.name, "कखग"[: 1 + i % 3]))
    csv = tmp_path / "t.csv"
    pd.DataFrame(rows, columns=["image_path", "text"]).to_csv(csv, index=False, encoding="utf-8")
    return csv


def test_item_and_collate(tmp_path):
    """One dataset item has the right fields, and collate_fn stacks images and pads labels with -100."""
    # Q: What does this test check? A: That the dataset has 10 items, that a grayscale file is returned as RGB, and that
    #    collate_fn stacks two samples into pixel_values (2, 3, 8, 8) and pads the short label row with -100.
    # Q: Why does it matter? What bug would make it fail? A: If load_image forgot convert("RGB"), grayscale files would
    #    break the 3-channel encoder. If collate_fn did not pad with -100 (or padded with a real id), the loss would
    #    train on padding. If labels did not end with <eos>, the model would never learn to stop.
    # Q: How is the input built and why is it enough? A: make_csv writes 10 fake images; ds[0] has text "क" (short) and
    #    ds[2] has "कखग" (long). One short and one long label is the smallest case that needs padding.
    # Q: Which project functions does it exercise? A: TrOCRWordDataset.__len__, load_image, __getitem__, collate_fn
    #    (src/trocr/dataset.py) and CharTokenizer.encode / decode (src/trocr/tokenizer.py).
    tok = CharTokenizer(list("कखग"))
    ds = TrOCRWordDataset(make_csv(tmp_path), tok, DummyProc(), root=str(tmp_path))
    # 10 rows were written, and the grayscale file must come back as RGB.
    assert len(ds) == 10 and ds.load_image(0).mode == "RGB"
    b = collate_fn([ds[0], ds[2]])  # texts "क" and "कखग"
    # Two images of shape (3, 8, 8) from the dummy processor -> batch (2, 3, 8, 8).
    assert b["pixel_values"].shape == (2, 3, 8, 8)
    # "क" -> [id(क), EOS], then padded with -100 up to the length of the longer label (4 = 3 letters + EOS).
    assert b["labels"][0].tolist() == [tok.stoi["क"], EOS, -100, -100]
    # The long label decodes back to the original text, and the raw texts are passed through unchanged.
    assert tok.decode(b["labels"][1]) == "कखग" and b["texts"] == ["क", "कखग"]


def test_first_n_and_random_n(tmp_path):
    """first_n keeps the first N rows; random_n keeps N random rows and is reproducible with the same seed."""
    # Q: What does this test check? A: first_n=4 keeps exactly 4 rows; random_n=5 keeps 5 rows, and building it twice
    #    with seed 42 picks the same 5 rows (the two .paths lists are equal).
    # Q: Why does it matter? What bug would make it fail? A: Reproducibility. If the seed were ignored, train and
    #    validation subsets would change on every run and results could not be compared. If the subset code were
    #    wrong, the overfit test (first_n) or the limited-budget run (random_n) would use the wrong number of images.
    # Q: How is the input built and why is it enough? A: The same 10-row fake CSV is used. 4 and 5 are smaller than 10,
    #    so the subset branches really run (random_n is only applied when it is smaller than the number of rows).
    # Q: Which project function does it exercise? A: TrOCRWordDataset.__init__ (the first_n / random_n / seed part).
    tok, csv = CharTokenizer(list("कखग")), make_csv(tmp_path)
    # Arguments shared by all datasets in this test.
    kw = dict(tokenizer=tok, image_processor=DummyProc(), root=str(tmp_path))
    assert len(TrOCRWordDataset(csv, first_n=4, **kw)) == 4
    # Build the same random subset twice with seed 42 and compare.
    a, b = (TrOCRWordDataset(csv, random_n=5, seed=42, **kw) for _ in range(2))
    assert len(a) == 5 and a.paths == b.paths  # seeded, reproducible


def test_missing_images_give_a_clear_error(tmp_path):
    """If the first image does not exist, the dataset must fail at once with a helpful message."""
    # Q: What does this test check? A: If the first image in the CSV does not exist, creating the dataset raises
    #    FileNotFoundError with a message that contains "first image not found".
    # Q: Why does it matter? What bug would make it fail? A: The data is large and is often not extracted yet or the
    #    command is run from the wrong folder. Without this early check, the crash would come later (when the first
    #    image is read) with a less helpful message. The test fails if the check is removed or the text changes.
    # Q: How is the input built and why is it enough? A: A one-row CSV points to data/raw/nowhere/1.jpg, which cannot
    #    exist under tmp_path. No image file is created at all, because a missing file is exactly the case.
    # Q: Which project function does it exercise? A: The os.path.exists check at the end of TrOCRWordDataset.__init__.
    import pytest

    csv = tmp_path / "t.csv"
    # The CSV points to a file that does not exist.
    pd.DataFrame([("data/raw/nowhere/1.jpg", "क")], columns=["image_path", "text"]).to_csv(csv, index=False, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="first image not found"):
        TrOCRWordDataset(csv, CharTokenizer(list("क")), DummyProc(), root=str(tmp_path))
