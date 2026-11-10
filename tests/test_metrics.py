import pytest

from src.common.metrics import accuracy, cer, wer


def test_cer_identical_is_zero():
    assert cer(["कमल"], ["कमल"]) == 0.0


def test_cer_one_substitution():
    # "कमल" has 3 code points, one substituted -> 1/3
    assert cer(["कमल"], ["कमन"]) == pytest.approx(1 / 3)


def test_cer_counts_matras_as_code_points():
    # "किताब" = क ि त ा ब (5 code points); dropping the matra ि is 1 deletion
    assert cer(["किताब"], ["कताब"]) == pytest.approx(1 / 5)


def test_cer_is_nfc_normalised():
    # U+0958 (क़ precomposed) normalises to क + nukta under NFC, so no error is counted
    assert cer(["क़"], ["क़"]) == 0.0


def test_cer_aggregates_over_all_samples():
    # 1 error over 3 + 3 reference code points
    assert cer(["कमल", "कमल"], ["कमल", "कमन"]) == pytest.approx(1 / 6)


def test_cer_empty_reference_does_not_crash():
    assert cer([""], [""]) == 0.0


def test_wer_counts_words():
    assert wer(["क ख ग"], ["क घ ग"]) == pytest.approx(1 / 3)


def test_accuracy():
    assert accuracy(["कमल", "नल"], ["कमल", "नर"]) == 0.5
