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

# ======================================================================================================
# CONCEPTS USED IN THIS FILE
# unit test: a test of one small function in isolation. Here the function is save_results (src/common/io.py).
# assert: a line that must be true; if it is false, pytest marks the test as failed.
# pytest.raises: passes only if the code inside the `with` block raises the named error (here ValueError).
#   match="params" also checks that the error message contains that word.
# fixture: a helper that pytest builds and passes in by argument name. tmp_path and monkeypatch are built in.
# tmp_path: a fresh empty temporary folder for each test, so no test files are left in the real project.
# monkeypatch / monkeypatch.chdir: a pytest tool that changes something only for one test and undoes it after.
#   chdir changes the working folder, because save_results writes to the relative path results/<run_name>.
# fake / synthetic data: two or three made-up words and file names such as "a.png". No images or models are used.
# CSV (comma-separated values): a plain text table. test_predictions.csv has the header image_path, ground_truth,
#   prediction and one row per image. csv.reader reads it back as lists of strings.
# JSON: a text format for key/value data. metrics.json holds run_name, num_samples, cer, wer, accuracy, extra keys.
# round trip: write a file, read it back, compare. test_save_results_writes_both_files does this for both files.
# Contract 4: the project's agreed output format, so all models (CNN, CRNN, TrOCR-style) write the same files
#   and scripts/make_results_table.py can compare them.
# REQUIRED keys (EXTRA above): model, train_subset, decoding, params, train_time_min, ms_per_image.
# CER (character error rate): edit distance divided by true length, counted on Unicode code points; 0 is perfect.
# accuracy: fraction of words that match exactly. Here 1 of 2 words matches, so 0.5.
# negative test: a test that gives bad input on purpose and checks that the code refuses it with a clear error.
# UTF-8: the text encoding needed to store Devanagari letters in a file; the test reads the files as utf-8.
# ======================================================================================================

# ======================================================================================================
# Q&A (FILE LEVEL)
# Q: Which part of the project does this file protect? A: The shared results writer save_results in
#    src/common/io.py, which every model calls at the end of evaluation to write test_predictions.csv and metrics.json.
# Q: How is it run? A: From the repository root: python -m pytest tests/test_io.py.
# Q: Why is no real dataset or model needed? A: save_results only takes lists of paths and strings, so a few
#    made-up words are enough to check the files and the errors. Nothing is predicted or trained.
# Q: Why do the tests change the working folder? A: save_results writes to a relative results/ folder, so the
#    tests move into tmp_path first. Otherwise they would create files in the real project.
# ======================================================================================================

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
    # Q: What does this test check? A: That save_results creates results/crnn_full_greedy/ with test_predictions.csv
    #    (header plus two rows in the right order) and metrics.json (accuracy 0.5, 2 samples, extra keys copied).
    # Q: Why does it matter? What bug would make it fail? A: If the header, column order or row order were wrong, the
    #    results table script and the report would read wrong data. A returned dict that differs from the saved file
    #    would also be a bug.
    # Q: How is the fake input built and why is it enough? A: Two image names and two words; the second prediction
    #    differs by one letter, so accuracy must be 0.5. Any real pipeline output has the same list-of-strings form.
    # Q: Which function of the project does it exercise? A: save_results in src/common/io.py, which also calls cer, wer
    #    and accuracy from src/common/metrics.py.
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
    # Q: What does this test check? A: If the extra dict lacks a required key (params), save_results raises ValueError
    #    and the message names that key.
    # Q: Why does it matter? What bug would make it fail? A: Every model must report the same six keys, or the results
    #    table would have gaps. Without this check a model could forget params and nobody would notice until the report.
    # Q: How is the fake input built and why is it enough? A: A copy of EXTRA with params removed. One missing key
    #    is enough to trigger the check, which compares REQUIRED with the keys given.
    # Q: Which function of the project does it exercise? A: The missing-keys check at the start of save_results.
    monkeypatch.chdir(tmp_path)
    # Copy of EXTRA without the key params.
    bad = {k: v for k, v in EXTRA.items() if k != "params"}
    with pytest.raises(ValueError, match="params"):
        save_results("x", ["a.png"], ["क"], ["क"], bad)


def test_save_results_rejects_length_mismatch(tmp_path, monkeypatch):
    """Proves: ValueError is raised when the lists have different lengths
    (1 image path but 2 true words).
    """
    # Q: What does this test check? A: If image_paths, refs and hyps have different lengths, save_results raises
    #    ValueError instead of writing files.
    # Q: Why does it matter? What bug would make it fail? A: zip() silently cuts the longer list, so a mismatch would
    #    save a wrong CSV and wrong scores without any warning. The explicit length check prevents that.
    # Q: How is the fake input built and why is it enough? A: One image path but two true words and one prediction.
    #    Any unequal lengths hit the same check.
    # Q: Which function of the project does it exercise? A: The length check in save_results (src/common/io.py).
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        save_results("x", ["a.png"], ["क", "ख"], ["क"], EXTRA)
