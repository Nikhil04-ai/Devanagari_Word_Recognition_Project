import csv
import os

import pytest

torch = pytest.importorskip("torch")
from PIL import Image, ImageDraw  # noqa: E402

from src.crnn.train import load_config, run  # noqa: E402

WORDS = ["कमल", "मत", "तक", "लम", "कत", "मल", "तल", "कल"]


def _make_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    chars = sorted({c for w in WORDS for c in w})
    (data / "vocab.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    rows = []
    for i, w in enumerate(WORDS):
        im = Image.new("L", (140, 48), 255)
        ImageDraw.Draw(im).rectangle([5 + 10 * i, 10, 25 + 10 * i, 38], fill=0)
        path = f"img{i}.png"
        im.save(tmp_path / path)
        rows.append((path, w))
    for name in ("train.csv", "val.csv"):
        with open(data / name, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["image_path", "text"])
            wr.writerows(rows)
    return data


def _cfg(tmp_path, data, epochs):
    return load_config(
        None, data_dir=str(data), image_root=str(tmp_path), epochs=epochs, batch_size=4,
        num_workers=0, ckpt_dir=str(tmp_path / "ckpt"), hidden=32)


def test_training_writes_checkpoints_and_resumes(tmp_path):
    data = _make_data(tmp_path)
    cfg = _cfg(tmp_path, data, epochs=3)
    cfg["max_minutes"] = 0.0005  # first run is cut short after epoch 0, like a Kaggle disconnect
    out = run(cfg)
    assert out["last_epoch"] == 0
    for name in ("last.pt", "best.pt", "log.csv"):
        assert os.path.exists(tmp_path / "ckpt" / name)

    out2 = run(_cfg(tmp_path, data, epochs=3), resume=True)  # same config: continues with epoch 1
    assert out2["last_epoch"] == 2
    rows = list(csv.reader(open(tmp_path / "ckpt" / "log.csv", encoding="utf-8")))
    assert len(rows) == 1 + 3  # header + 3 epochs


def test_time_budget_stops_training(tmp_path):
    data = _make_data(tmp_path)
    cfg = _cfg(tmp_path, data, epochs=50)
    cfg["max_minutes"] = 0.0005  # about 0.03 s
    out = run(cfg)
    assert out["last_epoch"] < 49
    assert os.path.exists(tmp_path / "ckpt" / "best.pt")  # a usable model is always saved
