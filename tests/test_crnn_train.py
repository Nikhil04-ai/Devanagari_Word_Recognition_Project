"""Tests for the CRNN training loop in src/crnn/train.py.

They train a very small model for a few epochs on a tiny made-up dataset (8 images), so they run
fast on a CPU. They check that training writes its checkpoint files, that it can resume after an
interruption, and that the time budget stops it cleanly.
Run with: pytest tests/test_crnn_train.py
(The whole file is skipped if PyTorch is not installed.)
"""
# csv = write the data files; os = check that files exist.
import csv
import os

# pytest = the test runner.
import pytest

# importorskip: if torch is not installed, skip every test in this file instead of failing.
torch = pytest.importorskip("torch")
# Pillow: creates the small test images. The noqa notes tell the style checker that these imports
# come after the torch line on purpose.
from PIL import Image, ImageDraw  # noqa: E402

# load_config builds the settings dict; run does the training.
from src.crnn.train import load_config, run  # noqa: E402

# Eight short fake words; their letters make the vocabulary.
WORDS = ["कमल", "मत", "तक", "लम", "कत", "मल", "तल", "कल"]


def _make_data(tmp_path):
    """Make a tiny dataset inside the temporary folder tmp_path and return its data folder.
    It writes vocab.txt (every letter once), 8 PNG images, and train.csv and val.csv.
    Both CSV files contain the same 8 rows, which is enough for these tests.
    """
    # Folder for vocab.txt and the CSV files.
    data = tmp_path / "data"
    data.mkdir()
    # Sorted list of the different letters used by all words.
    chars = sorted({c for w in WORDS for c in w})
    # One letter per line, the format CharVocab.from_file reads.
    (data / "vocab.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    rows = []
    # One image per word.
    for i, w in enumerate(WORDS):
        # White 140 x 48 image.
        im = Image.new("L", (140, 48), 255)
        # A black rectangle at a different x position for each word, so the images are not all the same.
        ImageDraw.Draw(im).rectangle([5 + 10 * i, 10, 25 + 10 * i, 38], fill=0)
        path = f"img{i}.png"
        # Image path is relative to tmp_path (the image_root used in the config).
        im.save(tmp_path / path)
        rows.append((path, w))
    # Write the same rows to train.csv and val.csv with the header image_path,text.
    for name in ("train.csv", "val.csv"):
        with open(data / name, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["image_path", "text"])
            wr.writerows(rows)
    return data


def _cfg(tmp_path, data, epochs):
    """Build a training config from the defaults plus a few overrides.
    Inputs: tmp_path (temporary folder), data (data folder), epochs (number of epochs).
    The path None means: use no YAML file, only the defaults in src/crnn/train.py.
    """
    return load_config(
        # batch_size 4 gives 2 updates per epoch for 8 images. image_root and ckpt_dir point into the
        # temporary folder. num_workers=0 loads data in the main process. hidden=32 makes the model small.
        None, data_dir=str(data), image_root=str(tmp_path), epochs=epochs, batch_size=4,
        num_workers=0, ckpt_dir=str(tmp_path / "ckpt"), hidden=32)


def test_training_writes_checkpoints_and_resumes(tmp_path):
    """Proves: training writes last.pt, best.pt and log.csv, and resuming continues where it stopped.
    Run 1 is stopped by a tiny time budget after epoch 0. Run 2 resumes and finishes epochs 1 and 2.
    """
    # Make the data and a config for 3 epochs.
    data = _make_data(tmp_path)
    cfg = _cfg(tmp_path, data, epochs=3)
    # 0.0005 minutes is about 0.03 seconds. The loop checks the budget after every batch, so the
    # run stops during epoch 0 (like a Kaggle session that disconnects).
    cfg["max_minutes"] = 0.0005  # first run is cut short after epoch 0, like a Kaggle disconnect
    # Run the training; it returns a dict with the last finished epoch.
    out = run(cfg)
    # Epoch 0 was still finished (validated and saved) before the loop stopped.
    assert out["last_epoch"] == 0
    # These three files must exist: the last checkpoint, the best checkpoint and the training log.
    for name in ("last.pt", "best.pt", "log.csv"):
        assert os.path.exists(tmp_path / "ckpt" / name)

    # A fresh config with the same settings but no time limit; resume=True loads last.pt.
    # The same epochs and data are needed because the learning-rate schedule has a fixed number of steps.
    out2 = run(_cfg(tmp_path, data, epochs=3), resume=True)  # same config: continues with epoch 1
    # It continued with epoch 1 and ended with epoch 2 (the last of 3 epochs, counted from 0).
    assert out2["last_epoch"] == 2
    # Read the log back as rows.
    rows = list(csv.reader(open(tmp_path / "ckpt" / "log.csv", encoding="utf-8")))
    # 1 header row + 3 epoch rows. If resume had restarted the log, there would be fewer rows.
    assert len(rows) == 1 + 3  # header + 3 epochs


def test_time_budget_stops_training(tmp_path):
    """Proves: max_minutes stops training early, and a usable model (best.pt) is always saved."""
    data = _make_data(tmp_path)
    # 50 epochs are asked for, but the budget is tiny.
    cfg = _cfg(tmp_path, data, epochs=50)
    # 0.0005 minutes is about 0.03 seconds.
    cfg["max_minutes"] = 0.0005  # about 0.03 s
    out = run(cfg)
    # Training stopped long before the last epoch (index 49).
    assert out["last_epoch"] < 49
    # Even a very short run has saved best.pt, because the first validation always counts as an improvement.
    assert os.path.exists(tmp_path / "ckpt" / "best.pt")  # a usable model is always saved
