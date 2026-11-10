"""Shared metrics (Contract 3). Every model reports its numbers through this file.

All strings are NFC-normalised before comparison, so the same Devanagari word
stored with different code-point sequences is never counted as an error.
"""
import unicodedata

from rapidfuzz.distance import Levenshtein


def _norm(s) -> str:
    return unicodedata.normalize("NFC", str(s).strip())


def cer(refs, hyps) -> float:
    """Character Error Rate over Unicode code points: total edit distance / total reference length."""
    errors = sum(Levenshtein.distance(_norm(r), _norm(h)) for r, h in zip(refs, hyps))
    total = sum(len(_norm(r)) for r in refs)
    return errors / max(total, 1)


def wer(refs, hyps) -> float:
    """Word Error Rate: edit distance over whitespace-separated words."""
    errors = sum(Levenshtein.distance(_norm(r).split(), _norm(h).split()) for r, h in zip(refs, hyps))
    total = sum(len(_norm(r).split()) for r in refs)
    return errors / max(total, 1)


def accuracy(refs, hyps) -> float:
    """Exact-match accuracy."""
    return sum(_norm(r) == _norm(h) for r, h in zip(refs, hyps)) / max(len(refs), 1)
