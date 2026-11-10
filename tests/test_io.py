import csv
import json

import pytest

from src.common.io import save_results

EXTRA = {
    "model": "crnn",
    "train_subset": "full",
    "decoding": "greedy",
    "params": 1234,
    "train_time_min": 1.5,
    "ms_per_image": 2.0,
}


def test_save_results_writes_both_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = save_results("crnn_full_greedy", ["a.png", "b.png"], ["कमल", "नल"], ["कमल", "नर"], EXTRA)

    out = tmp_path / "results" / "crnn_full_greedy"
    with open(out / "test_predictions.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["image_path", "ground_truth", "prediction"]
    assert rows[1] == ["a.png", "कमल", "कमल"]
    assert rows[2] == ["b.png", "नल", "नर"]

    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["accuracy"] == 0.5
    assert saved["num_samples"] == 2
    assert saved["model"] == "crnn"
    assert metrics["cer"] == saved["cer"]


def test_save_results_rejects_missing_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = {k: v for k, v in EXTRA.items() if k != "params"}
    with pytest.raises(ValueError, match="params"):
        save_results("x", ["a.png"], ["क"], ["क"], bad)


def test_save_results_rejects_length_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        save_results("x", ["a.png"], ["क", "ख"], ["क"], EXTRA)
