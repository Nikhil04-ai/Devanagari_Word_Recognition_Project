"""Dataset / collate tests with a dummy image processor (no downloads)."""
import types

import pandas as pd
import torch
from PIL import Image

from src.trocr.dataset import TrOCRWordDataset, collate_fn
from src.trocr.tokenizer import EOS, CharTokenizer


class DummyProc:
    def __call__(self, images, return_tensors="pt"):
        return types.SimpleNamespace(pixel_values=torch.zeros(1, 3, 8, 8))


def make_csv(tmp_path, n=10):
    rows = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.new("L", (30, 10), 255).save(p)  # grayscale on purpose: loader must convert to RGB
        rows.append((p.name, "कखग"[: 1 + i % 3]))
    csv = tmp_path / "t.csv"
    pd.DataFrame(rows, columns=["image_path", "text"]).to_csv(csv, index=False, encoding="utf-8")
    return csv


def test_item_and_collate(tmp_path):
    tok = CharTokenizer(list("कखग"))
    ds = TrOCRWordDataset(make_csv(tmp_path), tok, DummyProc(), root=str(tmp_path))
    assert len(ds) == 10 and ds.load_image(0).mode == "RGB"
    b = collate_fn([ds[0], ds[2]])  # texts "क" and "कखग"
    assert b["pixel_values"].shape == (2, 3, 8, 8)
    assert b["labels"][0].tolist() == [tok.stoi["क"], EOS, -100, -100]
    assert tok.decode(b["labels"][1]) == "कखग" and b["texts"] == ["क", "कखग"]


def test_first_n_and_random_n(tmp_path):
    tok, csv = CharTokenizer(list("कखग")), make_csv(tmp_path)
    kw = dict(tokenizer=tok, image_processor=DummyProc(), root=str(tmp_path))
    assert len(TrOCRWordDataset(csv, first_n=4, **kw)) == 4
    a, b = (TrOCRWordDataset(csv, random_n=5, seed=42, **kw) for _ in range(2))
    assert len(a) == 5 and a.paths == b.paths  # seeded, reproducible


def test_missing_images_give_a_clear_error(tmp_path):
    import pytest

    csv = tmp_path / "t.csv"
    pd.DataFrame([("data/raw/nowhere/1.jpg", "क")], columns=["image_path", "text"]).to_csv(csv, index=False, encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="first image not found"):
        TrOCRWordDataset(csv, CharTokenizer(list("क")), DummyProc(), root=str(tmp_path))
