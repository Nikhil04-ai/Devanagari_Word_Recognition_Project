"""Unit tests for the shared metrics in src/common/metrics.py (cer, wer, accuracy).

Every model reports its numbers through these functions, so a mistake here would make all results
wrong. Each test checks one rule with a tiny example whose answer can be worked out by hand.
Run with: pytest tests/test_metrics.py
"""
# pytest = the test runner; pytest.approx compares floating point numbers with a small tolerance.
import pytest

# The three metric functions under test.
from src.common.metrics import accuracy, cer, wer

# ======================================================================================================
# CONCEPTS USED IN THIS FILE
# unit test: a test of one small function in isolation. Here the functions are cer, wer and accuracy.
# regression test: a test that stays in the project so a bug that was fixed or ruled out cannot come back
#   unnoticed. These tests protect the numbers that every model reports.
# assert: a line that must be true; if it is false, pytest marks the test as failed.
# pytest.approx: compares floats with a small tolerance, because values like 1/3 cannot be stored exactly.
# synthetic / hand-made data: tiny words whose correct answer can be worked out by hand, so no dataset is used.
# determinism: the metrics have no randomness, so the same input always gives the same number. No seed is needed.
# Levenshtein distance (edit distance): the fewest insertions, deletions and substitutions to turn the prediction
#   into the true text. cer and wer use it (from the rapidfuzz library) in src/common/metrics.py.
# CER (character error rate): total edits / total true characters over the whole list. 0.0 is perfect and it can
#   be above 1.0. Used in the cer tests.
# WER (word error rate): the same idea, but the text is split at spaces and whole words are compared
#   (test_wer_counts_words).
# accuracy: exact-match rate, the share of predictions equal to the true text (test_accuracy).
# Unicode code point: one numbered symbol of Unicode. CER counts code points, not visual letters.
# matra: a Devanagari vowel sign such as "ि". It is its own code point, so "किताब" has 5 code points
#   (test_cer_counts_matras_as_code_points).
# nukta: a dot sign (U+093C) that can be stored as a separate code point. U+0958 is the pre-combined form of क़.
# NFC normalisation: Unicode makes text with the same look use the same code points, so identical-looking
#   words are never counted as errors (test_cer_is_nfc_normalised).
# micro-average / aggregation: add all edits and all true lengths first, then divide once. It is not the average
#   of per-word rates, so long words count more (test_cer_aggregates_over_all_samples).
# edge case / division by zero: an empty true text gives total length 0; the code divides by max(total, 1).
# ======================================================================================================

# ======================================================================================================
# Q&A (FILE LEVEL)
# Q: Which part of the project does this file protect? A: The shared metrics in src/common/metrics.py. io.py uses
#    them to write metrics.json and train.py uses cer and wer on the validation set, so all models use the same scores.
# Q: How is it run? A: From the repository root: python -m pytest tests/test_metrics.py. It is fast and needs
#    only pytest and rapidfuzz.
# Q: Why is no real dataset needed? A: Each test gives a few made-up words with a known answer, such as one wrong
#    letter in a 3-letter word giving 1/3. The metric functions only compare strings.
# Q: Why do these tests matter so much? A: If a metric were wrong, every reported result of every model would be
#    wrong, even if the models themselves were good.
# ======================================================================================================


def test_cer_identical_is_zero():
    """Proves: when the prediction equals the true word, CER is exactly 0."""
    # Q: What does this test check? A: When the prediction equals the true word, CER is exactly 0.0.
    # Q: Why does it matter? What bug would make it fail? A: A perfect model must score 0. A wrong normalisation or an
    #    off-by-one in the edit distance would give a non-zero error for correct text.
    # Q: How is the input built and why is it enough? A: One word "कमल" as truth and prediction. It is the simplest
    #    case; the edit distance of equal strings must be 0.
    # Q: Which function of the project does it exercise? A: cer in src/common/metrics.py.
    assert cer(["कमल"], ["कमल"]) == 0.0


def test_cer_one_substitution():
    """Proves: one wrong letter in a 3-letter word gives CER 1/3 (1 edit / 3 true characters)."""
    # Q: What does this test check? A: "कमल" predicted as "कमन" has one substituted letter, so CER = 1/3.
    # Q: Why does it matter? What bug would make it fail? A: It shows that one error is counted once and divided by
    #    the true length (3). Dividing by the wrong length or counting a substitution as two edits would change it.
    # Q: How is the input built and why is it enough? A: A 3-letter word with exactly one wrong letter, so the right
    #    answer is known by hand. pytest.approx is used because 1/3 is not exact as a float.
    # Q: Which function of the project does it exercise? A: cer in src/common/metrics.py.
    # "कमल" has 3 code points, one substituted -> 1/3
    # approx is needed because 1/3 cannot be stored exactly as a float.
    assert cer(["कमल"], ["कमन"]) == pytest.approx(1 / 3)


def test_cer_counts_matras_as_code_points():
    """Proves: CER counts Unicode code points, so a vowel sign (matra) counts as its own character.
    Deleting the matra from a 5-code-point word is 1 error, giving 1/5.
    """
    # Q: What does this test check? A: "किताब" (5 code points) predicted as "कताब" (matra "ि" missing) gives CER 1/5.
    # Q: Why does it matter? What bug would make it fail? A: The models output code points, so the metric must count
    #    them too. If CER counted visual letters (क+ि as one), the answer would differ, and results would be unfair.
    # Q: How is the input built and why is it enough? A: A real Devanagari word where the only difference is one
    #    vowel sign, which is the typical error and shows that a matra is its own character.
    # Q: Which function of the project does it exercise? A: cer in src/common/metrics.py.
    # "किताब" = क ि त ा ब (5 code points); dropping the matra ि is 1 deletion
    assert cer(["किताब"], ["कताब"]) == pytest.approx(1 / 5)


def test_cer_is_nfc_normalised():
    """Proves: strings are normalised to NFC before comparing.
    The first string uses the single code point U+0958; the second uses the letter plus a separate
    nukta sign. They look the same, and after NFC they are equal, so CER is 0.
    """
    # Q: What does this test check? A: The same-looking word written with the single code point U+0958 and with
    #    क + nukta (U+093C) gives CER 0.0, because both are NFC-normalised before comparing.
    # Q: Why does it matter? What bug would make it fail? A: Without normalisation, text that looks identical but is
    #    stored differently would be counted as an error, and a correct prediction would be punished.
    # Q: How is the input built and why is it enough? A: One string for each form of the letter. Under NFC, U+0958
    #    becomes क + nukta, so the two strings become equal. This is the classic Devanagari case.
    # Q: Which function of the project does it exercise? A: cer and its helper _norm in src/common/metrics.py.
    # U+0958 (क़ precomposed) normalises to क + nukta under NFC, so no error is counted
    assert cer(["क़"], ["क़"]) == 0.0


def test_cer_aggregates_over_all_samples():
    """Proves: CER is a total over all samples (sum of edits / sum of true lengths),
    not an average of per-sample rates.
    """
    # Q: What does this test check? A: With two words, one perfect and one with 1 edit, CER = 1 edit / (3+3) = 1/6.
    # Q: Why does it matter? What bug would make it fail? A: CER must be total edits over total true length, over all
    #    samples. A bug that scored only the last word (1/3) or divided by the number of words would fail. Note: both
    #    words have length 3, so a per-word average (0 and 1/3) would also give 1/6; this test does not tell them apart.
    # Q: How is the input built and why is it enough? A: Two words of equal length, so the answer is easy by hand.
    # Q: Which function of the project does it exercise? A: cer in src/common/metrics.py (sum of edits, sum of lengths).
    # 1 error over 3 + 3 reference code points
    # First pair is perfect, second has 1 edit: 1 error over 3 + 3 true code points = 1/6.
    assert cer(["कमल", "कमल"], ["कमल", "कमन"]) == pytest.approx(1 / 6)


def test_cer_empty_reference_does_not_crash():
    """Proves: an empty true word does not cause a division by zero (the code divides by max(total, 1))."""
    # Q: What does this test check? A: An empty true word and an empty prediction give CER 0.0 and do not crash.
    # Q: Why does it matter? What bug would make it fail? A: The total true length is 0 here. Without the
    #    max(total, 1) guard, cer would raise ZeroDivisionError on an empty reference.
    # Q: How is the input built and why is it enough? A: One empty string for each side; it is the smallest case that
    #    makes the total length 0.
    # Q: Which function of the project does it exercise? A: cer in src/common/metrics.py (the max(total, 1) guard).
    assert cer([""], [""]) == 0.0


def test_wer_counts_words():
    """Proves: WER works on whole words split at spaces. One wrong word out of three gives 1/3."""
    # Q: What does this test check? A: For "क ख ग" against "क घ ग", one of three words is wrong, so WER = 1/3.
    # Q: Why does it matter? What bug would make it fail? A: WER must compare whole words split at spaces. If it
    #    compared characters (like CER), the answer would be based on character counts, not words.
    # Q: How is the input built and why is it enough? A: A sentence of three one-letter words with one changed, so
    #    the number of words and the number of errors are clear by hand.
    # Q: Which function of the project does it exercise? A: wer in src/common/metrics.py.
    assert wer(["क ख ग"], ["क घ ग"]) == pytest.approx(1 / 3)


def test_accuracy():
    """Proves: accuracy is exact match. One of the two words matches, so the result is 0.5."""
    # Q: What does this test check? A: With two words where only the first prediction is exactly right, accuracy is 0.5.
    # Q: Why does it matter? What bug would make it fail? A: Accuracy is exact match only. A near miss like "नर" for
    #    "नल" must count as wrong, so a bug that gave partial credit would fail this test.
    # Q: How is the input built and why is it enough? A: One exact match and one single-letter error give a result
    #    (1 of 2) that can be checked by hand. 0.5 is exact in a float, so no approx is needed.
    # Q: Which function of the project does it exercise? A: accuracy in src/common/metrics.py.
    assert accuracy(["कमल", "नल"], ["कमल", "नर"]) == 0.5
