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


class DummyProc:
    """Fake image processor: ignores the image and always returns a zero tensor of shape (1, 3, 8, 8).
    It has the same call signature as the real processor, so the dataset code runs unchanged."""

    def __call__(self, images, return_tensors="pt"):
        return types.SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8))


def make_csv(tmp_path, n=10):
    """Create n small image files and a CSV manifest in the pytest temp folder. Returns the CSV path.
    Row i has the text "क", "कख" or "कखग" (length 1, 2, 3, repeating), so label lengths differ."""
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
    tok, csv = CharTokenizer(list("कखग")), make_csv(tmp_path)
    # Arguments shared by all datasets in this test.
    kw = dict(tokenizer=tok, image_processor=DummyProc(), root=str(tmp_path))
    assert len(TrOCRWordDataset(csv, first_n=4, **kw)) == 4
    # Build the same random subset twice with seed 42 and compare.
    a, b = (TrOCRWordDataset(csv, random_n=5, seed=42, **kw) for _ in range(2))
    assert len(a) == 5 and a.paths == b.paths  # seeded, reproducible


def test_missing_images_give_a_clear_error(tmp_path):
    """If the first image does not exist, the dataset must fail at once with a helpful message."""
    import pytest

    csv = tmp_path / "t.csv"
    # The CSV points to a file that does not exist.
    pd.DataFrame([("data/raw/nowhere/1.jpg", "क")], columns=["image_path", "text"]).to_csv(csv, index=False, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="first image not found"):
        TrOCRWordDataset(csv, CharTokenizer(list("क")), DummyProc(), root=str(tmp_path))
