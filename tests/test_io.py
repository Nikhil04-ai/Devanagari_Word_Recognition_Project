"""Unit tests for save_results in src/common/io.py.

save_results is the one shared function that every model uses to write its results:
results/<run_name>/test_predictions.csv and results/<run_name>/metrics.json.
These tests check the files it writes and the errors it raises for bad input.
Run with: pytest tests/test_io.py
"""
# csv and json = read back the files that save_results wrote.
import csv
import json

# pytest = the test runner (here used for pytest.raises).
import pytest

# The function under test.
from src.common.io import save_results

# Example of the extra information every run must give. These are the six keys that
# save_results requires: model name, how much training data was used, decoding method,
# number of parameters, training time (minutes) and prediction time per image (milliseconds).
EXTRA = {
    "model": "crnn",
    "train_subset": "full",
    "decoding": "greedy",
    "params": 1234,
    "train_time_min": 1.5,
    "ms_per_image": 2.0,
}


def test_save_results_writes_both_files(tmp_path, monkeypatch):
    """Proves: save_results writes both output files with the right content and returns the metrics.
    tmp_path is a fresh temporary folder given by pytest.
    """
    # Change the working folder to the temporary one. save_results writes to a relative results/ folder,
    # so this keeps the test files out of the real project.
    monkeypatch.chdir(tmp_path)
    # Two samples: the first prediction is right, the second is wrong in one letter.
    metrics = save_results("crnn_full_greedy", ["a.png", "b.png"], ["कमल", "नल"], ["कमल", "नर"], EXTRA)

    # Folder where save_results must have put the files.
    out = tmp_path / "results" / "crnn_full_greedy"
    # Read the predictions CSV back as a list of rows.
    with open(out / "test_predictions.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    # The first row must be the header.
    assert rows[0] == ["image_path", "ground_truth", "prediction"]
    # Data rows keep the order: image path, true word, predicted word.
    assert rows[1] == ["a.png", "कमल", "कमल"]
    assert rows[2] == ["b.png", "नल", "नर"]

    # Read metrics.json back. 1 of 2 words is exact, so accuracy is 0.5 and there are 2 samples.
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["accuracy"] == 0.5
    assert saved["num_samples"] == 2
    # Extra keys such as model are copied into the file.
    assert saved["model"] == "crnn"
    # The returned dict has the same CER value as the saved file.
    assert metrics["cer"] == saved["cer"]


def test_save_results_rejects_missing_keys(tmp_path, monkeypatch):
    """Proves: a missing required key (here params) raises ValueError, and the message names the key."""
    monkeypatch.chdir(tmp_path)
    # Copy of EXTRA without the key params.
    bad = {k: v for k, v in EXTRA.items() if k != "params"}
    with pytest.raises(ValueError, match="params"):
        save_results("x", ["a.png"], ["क"], ["क"], bad)


def test_save_results_rejects_length_mismatch(tmp_path, monkeypatch):
    """Proves: ValueError is raised when the lists have different lengths
    (1 image path but 2 true words).
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        save_results("x", ["a.png"], ["क", "ख"], ["क"], EXTRA)
