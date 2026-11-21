"""Unit tests for the shared metrics in src/common/metrics.py (cer, wer, accuracy).

Every model reports its numbers through these functions, so a mistake here would make all results
wrong. Each test checks one rule with a tiny example whose answer can be worked out by hand.
Run with: pytest tests/test_metrics.py
"""
# pytest = the test runner; pytest.approx compares floating point numbers with a small tolerance.
import pytest

# The three metric functions under test.
from src.common.metrics import accuracy, cer, wer


def test_cer_identical_is_zero():
    """Proves: when the prediction equals the true word, CER is exactly 0."""
    assert cer(["कमल"], ["कमल"]) == 0.0


def test_cer_one_substitution():
    """Proves: one wrong letter in a 3-letter word gives CER 1/3 (1 edit / 3 true characters)."""
    # "कमल" has 3 code points, one substituted -> 1/3
    # approx is needed because 1/3 cannot be stored exactly as a float.
    assert cer(["कमल"], ["कमन"]) == pytest.approx(1 / 3)


def test_cer_counts_matras_as_code_points():
    """Proves: CER counts Unicode code points, so a vowel sign (matra) counts as its own character.
    Deleting the matra from a 5-code-point word is 1 error, giving 1/5.
    """
    # "किताब" = क ि त ा ब (5 code points); dropping the matra ि is 1 deletion
    assert cer(["किताब"], ["कताब"]) == pytest.approx(1 / 5)


def test_cer_is_nfc_normalised():
    """Proves: strings are normalised to NFC before comparing.
    The first string uses the single code point U+0958; the second uses the letter plus a separate
    nukta sign. They look the same, and after NFC they are equal, so CER is 0.
    """
    # U+0958 (क़ precomposed) normalises to क + nukta under NFC, so no error is counted
    assert cer(["क़"], ["क़"]) == 0.0


def test_cer_aggregates_over_all_samples():
    """Proves: CER is a total over all samples (sum of edits / sum of true lengths),
    not an average of per-sample rates.
    """
    # 1 error over 3 + 3 reference code points
    # First pair is perfect, second has 1 edit: 1 error over 3 + 3 true code points = 1/6.
    assert cer(["कमल", "कमल"], ["कमल", "कमन"]) == pytest.approx(1 / 6)


def test_cer_empty_reference_does_not_crash():
    """Proves: an empty true word does not cause a division by zero (the code divides by max(total, 1))."""
    assert cer([""], [""]) == 0.0


def test_wer_counts_words():
    """Proves: WER works on whole words split at spaces. One wrong word out of three gives 1/3."""
    assert wer(["क ख ग"], ["क घ ग"]) == pytest.approx(1 / 3)


def test_accuracy():
    """Proves: accuracy is exact match. One of the two words matches, so the result is 0.5."""
    assert accuracy(["कमल", "नल"], ["कमल", "नर"]) == 0.5
