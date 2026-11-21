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

# ===== CONCEPTS USED IN THIS FILE =====
# CER (Character Error Rate): total character edits / total true characters. Lower is better. See cer().
# WER (Word Error Rate): the same idea with words as the units. See wer().
# Levenshtein / edit distance: the fewest insertions, deletions and substitutions to change one sequence into
#    another. rapidfuzz computes it. It works on strings and on lists of words.
# Insertion, deletion, substitution: the three kinds of edit. Each costs 1.
# Exact-match accuracy: the fraction of predictions identical to the true text. See accuracy().
# Unicode NFC: a standard form so the same Devanagari word always has the same code points. See _norm().
# Code point: one Unicode character unit. CER counts code points, which is what the model outputs.
# Corpus-level rate: errors and lengths are summed over the whole set before dividing, so long words count more.
# Reference vs hypothesis: refs = the true texts, hyps = the model's predictions (same order).

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: The shared scoring code. Every model (CNN, CRNN, TrOCR-style) is measured by these same three functions.
# Q: How is it run?
# A: Not directly. train.py calls cer and wer on the validation set, and io.py calls all three for the test results.
# Q: What does it read and write?
# A: It reads two lists of strings (true and predicted) and returns a number. It touches no files.
# Q: Why one shared file for metrics?
# A: If each model computed CER its own way, the numbers could not be compared. One file keeps the comparison fair.


def _norm(s) -> str:
    """Return s as a stripped, NFC-normalised string.

    str() so numbers or None-like values do not crash. strip() removes spaces and line breaks
    at the ends. NFC makes visually identical Devanagari text use identical code points.
    """
    # Q: What does it do and why does it exist?
    # A: It cleans a text before comparison: str(), strip, and NFC. cer, wer and accuracy all call it first.
    # Q: Why normalise to NFC?
    # A: The same Devanagari word can be stored with different code-point orders. Without NFC, such a word
    #    would be counted as an error even though it looks identical.
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
    # Q: What does it do and why does it exist?
    # A: It computes CER, the main score of the project: how many character edits are needed, divided by the
    #    number of true characters. It is 0.0 for a perfect result.
    # Q: Why sum errors first and divide once, and why max(total, 1)?
    # A: Summing first gives one rate for the whole set, where each character counts equally. max(total, 1)
    #    avoids a division by zero when the list is empty.
    # Sum of edit distances over all word pairs.
    errors = sum(Levenshtein.distance(_norm(r), _norm(h)) for r, h in zip(refs, hyps))
    total = sum(len(_norm(r)) for r in refs)
    return errors / max(total, 1)  # max(..., 1) avoids dividing by zero for an empty list


def wer(refs, hyps) -> float:
    """Word Error Rate: edit distance over whitespace-separated words.

    Same as cer, but each string is split into words first, so the edit distance counts words.
    Returns a float. For single-word images this is 0 or 1 per image.
    """
    # Q: What does it do and why does it exist?
    # A: It computes WER: the edit distance between the word lists, divided by the number of true words.
    #    It shows how often a whole word is wrong, which is stricter than CER.
    # Q: Why split into words before the distance?
    # A: Levenshtein.distance accepts lists, so after split() each word counts as one unit. For a single-word
    #    image a wrong prediction is one error out of one word, so WER is 0 or 1 per image.
    # Levenshtein.distance also works on lists, so it compares lists of words here.
    errors = sum(Levenshtein.distance(_norm(r).split(), _norm(h).split()) for r, h in zip(refs, hyps))
    total = sum(len(_norm(r).split()) for r in refs)
    return errors / max(total, 1)


def accuracy(refs, hyps) -> float:
    """Exact-match accuracy.

    Fraction of predictions that equal the true string exactly (after normalisation). Returns a float in [0, 1].
    """
    # Q: What does it do and why does it exist?
    # A: It returns the fraction of predictions that match the truth exactly. It is an easy number to explain
    #    and is saved next to CER and WER in metrics.json.
    # Q: Why normalise both sides and use max(len(refs), 1)?
    # A: Normalising stops code-point order differences from counting as mistakes. max(..., 1) avoids dividing
    #    by zero for an empty list.
    # True counts as 1 in sum(), so this counts the exact matches.
    return sum(_norm(r) == _norm(h) for r, h in zip(refs, hyps)) / max(len(refs), 1)
