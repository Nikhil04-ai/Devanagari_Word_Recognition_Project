"""Shared metrics (Contract 3). Every model reports its numbers through this file.

All strings are NFC-normalised before comparison, so the same Devanagari word
stored with different code-point sequences is never counted as an error.

Where this file fits: io.py calls cer, wer and accuracy to write metrics.json, and
train.py calls cer and wer on the validation set. Using one file means every model
(CNN, CRNN, TrOCR-style) is scored in exactly the same way.
"""
# unicodedata: NFC normalisation. rapidfuzz: fast Levenshtein (edit distance) in C++.
import unicodedata

from rapidfuzz.distance import Levenshtein


def _norm(s) -> str:
    """Return s as a stripped, NFC-normalised string.

    str() so numbers or None-like values do not crash. strip() removes spaces and line breaks
    at the ends. NFC makes visually identical Devanagari text use identical code points.
    """
    return unicodedata.normalize("NFC", str(s).strip())


def cer(refs, hyps) -> float:
    """Character Error Rate over Unicode code points: total edit distance / total reference length.

    refs: list of true strings. hyps: list of predicted strings, same order and length.
    Returns a float (0.0 = perfect, can be above 1.0).
    Edit distance = fewest insertions, deletions and substitutions to turn the prediction into the truth.
    It is counted on code points (not on visual letters), because that is what the model outputs.
    Total errors are divided by total characters over the whole set (not an average of per-word
    rates), so long words count more.
    """
    # Sum of edit distances over all word pairs.
    errors = sum(Levenshtein.distance(_norm(r), _norm(h)) for r, h in zip(refs, hyps))
    total = sum(len(_norm(r)) for r in refs)
    return errors / max(total, 1)  # max(..., 1) avoids dividing by zero for an empty list


def wer(refs, hyps) -> float:
    """Word Error Rate: edit distance over whitespace-separated words.

    Same as cer, but each string is split into words first, so the edit distance counts words.
    Returns a float. For single-word images this is 0 or 1 per image.
    """
    # Levenshtein.distance also works on lists, so it compares lists of words here.
    errors = sum(Levenshtein.distance(_norm(r).split(), _norm(h).split()) for r, h in zip(refs, hyps))
    total = sum(len(_norm(r).split()) for r in refs)
    return errors / max(total, 1)


def accuracy(refs, hyps) -> float:
    """Exact-match accuracy.

    Fraction of predictions that equal the true string exactly (after normalisation). Returns a float in [0, 1].
    """
    # True counts as 1 in sum(), so this counts the exact matches.
    return sum(_norm(r) == _norm(h) for r, h in zip(refs, hyps)) / max(len(refs), 1)
