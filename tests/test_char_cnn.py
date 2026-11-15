import csv
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from src.char_cnn.dataset import CLASSES, CHARS, NUM_CLASSES, load_split, random_affine, stratified_split, to_input  # noqa: E402
from src.char_cnn.model import CharCNN  # noqa: E402
from src.char_cnn.predict import load_model, predict, to_dhcd  # noqa: E402
from src.char_cnn.train import load_config, run  # noqa: E402


def _make_dhcd(root, per_class=3):
    """A tiny fake DHCD: every class gets a few 32x32 images with a different white bar."""
    for split in ("Train", "Test"):
        for idx, cls in enumerate(CLASSES):
            d = root / "DevanagariHandwrittenCharacterDataset" / split / cls
            d.mkdir(parents=True)
            for k in range(per_class):
                a = np.zeros((32, 32), dtype=np.uint8)
                a[2 + (idx % 23): 6 + (idx % 23), 2 + idx // 23 * 12: 14 + idx // 23 * 12 + k] = 255
                Image.fromarray(a).save(d / f"{k}.png")
    return root


def test_mapping_is_complete():
    assert NUM_CLASSES == 46 == len(set(CHARS))
    assert CHARS[0] == "क" and CHARS[35] == "ज्ञ" and CHARS[36] == "०"


def test_load_split_and_cache(tmp_path):
    root = _make_dhcd(tmp_path)
    x, y, paths = load_split(str(root), "Train")
    assert x.shape == (46 * 3, 32, 32) and x.dtype == torch.uint8
    assert sorted(set(y.tolist())) == list(range(46))
    assert os.path.exists(root / "cache_train.npz")
    x2, y2, paths2 = load_split(str(root), "Train")  # from the cache
    assert torch.equal(x, x2) and torch.equal(y, y2) and paths == paths2


def test_load_from_zip_matches_folders(tmp_path):
    import shutil

    root = _make_dhcd(tmp_path / "folders")
    x, y, _ = load_split(str(root), "Test", use_cache=False)
    zdir = tmp_path / "zipped"
    zdir.mkdir()
    shutil.make_archive(str(zdir / "dhcd"), "zip", root)
    xz, yz, pz = load_split(str(zdir), "Test", use_cache=False)
    assert torch.equal(x, xz) and torch.equal(y, yz)
    assert pz[0].endswith(".png") and "dhcd.zip/" in pz[0]


def test_stratified_split_keeps_every_class():
    labels = np.repeat(np.arange(46), 10)
    tr, va = stratified_split(labels, 0.1, seed=42)
    assert len(va) == 46 and len(set(labels[va])) == 46
    assert not set(tr) & set(va)


def test_model_and_augmentation_shapes():
    x = to_input(torch.randint(0, 255, (4, 32, 32), dtype=torch.uint8))
    assert random_affine(x).shape == x.shape
    assert CharCNN()(x).shape == (4, 46)


def test_train_resume_and_predict(tmp_path):
    root = _make_dhcd(tmp_path)
    cfg = load_config(None, data_dir=str(root), epochs=3, batch_size=32, val_fraction=0.34,
                      ckpt_dir=str(tmp_path / "ckpt"))
    cfg["max_minutes"] = 0.0001  # cut short after epoch 0, like a Kaggle disconnect
    out = run(cfg)
    assert out["last_epoch"] == 0
    for name in ("last.pt", "best.pt", "log.csv"):
        assert os.path.exists(tmp_path / "ckpt" / name)

    cfg["max_minutes"] = 0
    out2 = run(cfg, resume=True)
    assert out2["last_epoch"] == 2
    rows = list(csv.reader(open(tmp_path / "ckpt" / "log.csv", encoding="utf-8")))
    assert len(rows) == 1 + 3

    m = load_model(str(tmp_path / "ckpt" / "best.pt"))
    img = Image.new("RGB", (80, 60), "white")  # dark-on-light input of another size
    assert predict(m, img) in CHARS


def test_to_dhcd_inverts_and_centres():
    a = np.full((50, 70), 255, dtype=np.uint8)
    a[10:20, 30:40] = 0  # black ink on white paper
    out = to_dhcd(Image.fromarray(a))
    assert out.shape == (32, 32)
    assert out[16, 16] > 200 and out[0, 0] == 0  # ink is white and centred, border is black
