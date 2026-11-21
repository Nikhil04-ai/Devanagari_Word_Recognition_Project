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

# =====================================================================================================
# CONCEPTS USED IN THIS FILE
# =====================================================================================================
# unit test vs smoke / integration test: a unit test checks one small function. These two tests run the
#     whole training loop (data -> model -> loss -> save), so they are closer to a "smoke test":
#     they prove that the parts work together without crashing and that the files appear.
# regression test: a test kept so that a fixed bug (for example "resume restarts the log") cannot return.
# pytest: the test runner. It runs every function named test_* in files named test_*.py.
# assert: a line that must be true; if it is false, the test FAILS.
# pytest.importorskip("torch"): skips the whole file when PyTorch is not installed.
# fixture / tmp_path: tmp_path is a built-in pytest fixture. It gives each test its own fresh temporary
#     folder. All data and checkpoints are written there, so the real project files are never touched.
# fake / synthetic data: made-up input. Here: 8 small PNG images (one black rectangle on white) and 8
#     fake words. The model cannot really read them, but the training code does not care.
# determinism / seed: the training run calls set_seed(cfg["seed"]) (default 42) so random numbers
#     (weights, shuffling) repeat. These tests do not compare exact numbers, only epochs and files,
#     so they pass for any random values.
# epoch: one full pass over the training data. batch: a group of samples processed together.
#     Here 8 images with batch_size 4 give 2 batches (= 2 weight updates) per epoch.
# config (cfg): a dict of settings (epochs, batch_size, paths ...). load_config merges the defaults,
#     an optional YAML file and keyword overrides. The tests pass None for the file, so only defaults
#     plus overrides are used.
# num_workers=0: load data in the main process; simpler and safer in tests (no extra processes).
# hidden=32: a small LSTM size, so the network is light and the tests are fast on a CPU.
# vocabulary / vocab.txt: the file with one character per line; CharVocab.from_file reads it.
# CTC, blank symbol: the loss and the extra index 0 that the model uses for "no character here".
#     Training minimises the CTC loss between the model output and the true words.
# CER (character error rate): errors per character between predicted and true text; lower is better.
#     The best.pt file is saved whenever the validation CER improves.
# validation: after each epoch the model is scored on val.csv (here the same 8 rows as train.csv).
# checkpoint: a saved file with the model weights. last.pt (every epoch, also optimiser and schedule
#     state, so training can continue) and best.pt (the best validation CER, used later for testing).
# resume: continue an interrupted run from last.pt instead of starting again. Needed on Kaggle, where
#     a session can disconnect.
# learning-rate schedule (OneCycleLR): the learning rate rises and then falls over a fixed total number
#     of steps. This is why a resumed run must use the same epochs and data (same total steps).
# time budget (max_minutes): training stops cleanly after this many minutes; the epoch is still
#     validated and saved first. The tests use 0.0005 minutes (about 0.03 s) to force an early stop.
# log.csv: one row per epoch (loss, CER, WER, learning rate, minutes), plus one header row.
# early stopping / patience: stop if CER has not improved for `patience` epochs (default 8). It is not
#     triggered in these tests. overfit_n (overfit sanity check, learn a few samples to CER 0) is a
#     different mode of train.py and is not used in this file.
# =====================================================================================================
# Q&A (whole file)
# =====================================================================================================
# Q: What part of the project does this file protect?
# A: The training loop in src/crnn/train.py: load_config and run. It checks the checkpoint and log files,
#    resume after an interruption, and the time-budget stop.
# Q: How is it run?
# A: From the repository root with: pytest tests/test_crnn_train.py
# Q: Why is no real dataset needed?
# A: _make_data writes 8 tiny images, a vocab file and two CSV files into a temporary folder. The tests
#    check that the training machinery works, not that the model is accurate, so fake data is enough
#    and a few epochs on a CPU finish quickly.
# Q: Why are these tests important for a Kaggle run?
# A: Kaggle sessions have a time limit and can disconnect. If resume or the time-budget stop were broken,
#    hours of training could be lost or end without a usable best.pt.

# Eight short fake words; their letters make the vocabulary.
WORDS = ["कमल", "मत", "तक", "लम", "कत", "मल", "तल", "कल"]


def _make_data(tmp_path):
    """Make a tiny dataset inside the temporary folder tmp_path and return its data folder.
    It writes vocab.txt (every letter once), 8 PNG images, and train.csv and val.csv.
    Both CSV files contain the same 8 rows, which is enough for these tests.
    """
    # Q: What does this helper do?
    # A: It builds a tiny fake dataset inside tmp_path: data/vocab.txt, 8 PNG images, and data/train.csv and
    #    data/val.csv with the same 8 rows. It returns the data folder.
    # Q: Why is this enough for the tests?
    # A: train.py only needs a vocab file, two CSV files (image_path,text) and the images they point to.
    #    Each image has a rectangle at a different place, so the images differ, but real handwriting is
    #    not needed because the tests check files and epochs, not accuracy.
    # Q: Why are train.csv and val.csv the same?
    # A: Only the training machinery is tested, so using the same 8 rows is simplest. Real runs use
    #    separate train and validation files.
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
    # Q: What does this helper do?
    # A: It returns a training config: the defaults of src/crnn/train.py plus overrides that point to the
    #    temporary folder and make the run small (batch 4, hidden 32, num_workers 0).
    # Q: Why is a fresh config built for each run, even for resume?
    # A: run() reads the settings from the dict. A new dict with the same values means the resumed run
    #    uses the same epochs and data as the first run (needed by the learning-rate schedule), but
    #    without the max_minutes limit that the first run had.
    # Q: Which project code does it exercise?
    # A: load_config in src/crnn/train.py (path None = no YAML file).
    return load_config(
        # batch_size 4 gives 2 updates per epoch for 8 images. image_root and ckpt_dir point into the
        # temporary folder. num_workers=0 loads data in the main process. hidden=32 makes the model small.
        None, data_dir=str(data), image_root=str(tmp_path), epochs=epochs, batch_size=4,
        num_workers=0, ckpt_dir=str(tmp_path / "ckpt"), hidden=32)


def test_training_writes_checkpoints_and_resumes(tmp_path):
    """Proves: training writes last.pt, best.pt and log.csv, and resuming continues where it stopped.
    Run 1 is stopped by a tiny time budget after epoch 0. Run 2 resumes and finishes epochs 1 and 2.
    """
    # Q: What does this test check?
    # A: That a run cut short by the time budget has finished epoch 0 and written last.pt, best.pt and
    #    log.csv, and that a second run with resume=True continues with epochs 1 and 2 and keeps appending
    #    to the same log (1 header + 3 epoch rows).
    # Q: Why does it matter? What bug would make it fail?
    # A: If resume did not load last.pt, it would start again at epoch 0 and last_epoch would not be 2. If
    #    resume overwrote log.csv, the log would have fewer than 4 rows. If the saved schedule did not
    #    match the total steps, run() would raise an error. All of these would waste a Kaggle session.
    # Q: How is the input built and why is it enough?
    # A: _make_data gives 8 tiny images and _cfg a small 3-epoch config. max_minutes = 0.0005 stops the first
    #    run after its first batch, which imitates a disconnect. Epoch 0 is still validated and saved
    #    before the loop ends, which is what the test relies on.
    # Q: Which project code does it exercise?
    # A: run in src/crnn/train.py: checkpoint saving, the resume branch, the log file and the time-budget stop.
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
    # Q: What does this test check?
    # A: That with a tiny max_minutes the run stops long before the requested 50 epochs, and that best.pt
    #    still exists afterwards.
    # Q: Why does it matter? What bug would make it fail?
    # A: If the budget were ignored, training would run all 50 epochs and exceed the session time. If the
    #    run stopped without saving, there would be no usable model. The budget check is made after every
    #    batch, and validation and saving still happen before the loop ends.
    # Q: Why does best.pt always exist after even a very short run?
    # A: best_cer starts at infinity, so the first validated epoch always counts as an improvement and
    #    best.pt is written.
    # Q: Which project code does it exercise?
    # A: run in src/crnn/train.py: the max_minutes check, the time_up stop rule and torch.save of best.pt.
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
