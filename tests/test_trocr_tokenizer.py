"""Tests for CharTokenizer (src/trocr/tokenizer.py).

What is tested: fixed special-token ids, encode / decode round trip on real Devanagari words,
unknown characters and truncation, -100 padding in batch_encode, and saving / loading.
Run with: pytest tests/test_trocr_tokenizer.py
"""
# unicodedata: NFC normalisation, the same as the tokenizer uses.
import unicodedata

import torch

# The code under test and the special-token ids.
from src.trocr.tokenizer import BOS, EOS, PAD, UNK, CharTokenizer

# ======================================================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================================================
# unit test: a small function that runs one piece of code on tiny input and checks the result. Every function whose
#     name starts with test_ is one unit test; pytest finds and runs them.
# assert: a line that must be true. If it is false, the test fails. Used in every test here.
# fixture / tmp_path: a fixture is a helper object pytest passes in by argument name. tmp_path is the built-in one
#     that gives a fresh temporary folder; test_save_load and test_vocab_file write their files there.
# synthetic data: the WORDS list below is typed in by hand; no dataset file is read.
# tokenizer: converts text to integer ids (encode) and back (decode). The model only works with numbers.
# character-level tokenizer: one token per Unicode code point (letter, vowel sign / matra, virama). CharTokenizer
#     is built from a list of characters; make() builds it from the characters of WORDS.
# vocabulary: the set of known tokens. Here: 4 special tokens + the distinct characters. len(tok) is its size.
# special tokens: <pad>=0 (filler), <bos>=1 ("beginning of sequence", first decoder input), <eos>=2 ("end of
#     sequence", marks the word end), <unk>=3 (unknown character). Real characters start at id 4.
# stoi / itos: "string to id" and "id to string" tables of the tokenizer (stoi is a dict, itos is a list).
# encode / decode: encode(word) -> ids ending with <eos>; decode(ids) -> text, stopping at the first <eos>.
# round trip: decode(encode(w)) == w. If it holds for hard words, no information is lost by the tokenizer.
# Devanagari code points: conjuncts such as क्ष or ज्ञ are several code points joined by the virama (्).
#     So a word like "विद्यालय" has more code points than visible letters; WORDS covers these cases.
# NFC normalisation: unicodedata.normalize("NFC", s) gives one standard code-point form for the same visible text.
#     The tokenizer applies it in encode and decode, and the tests apply it to the expected word.
# unknown character (<unk>): a character that is not in the vocabulary is mapped to id 3 instead of crashing.
# truncation / max_length: encode cuts a long word so the result, including <eos>, has at most max_length ids.
# padding with -100 (IGNORE_INDEX): batch_encode fills short rows with -100. PyTorch cross-entropy ignores -100,
#     so padding adds no loss. decode() also skips -100.
# LongTensor: an integer tensor (torch.long); labels must be integers.
# batch: several words processed together as one (B, L) tensor; B = number of words, L = longest encoded word.
# JSON save / load: save writes the character list to a .json file, load rebuilds the tokenizer from it.
# vocab file: a text file with one character per line (vocab.txt); from_vocab_file reads it.

# ======================================================================================================
# Q&A ABOUT THIS FILE
# ======================================================================================================
# Q: What part of the project does this file protect? A: CharTokenizer in src/trocr/tokenizer.py. Every label
#    that reaches the TrOCR model and every predicted word shown to the user goes through it, so a bug here
#    would silently damage training and evaluation.
# Q: How is it run? A: From the repository root with: pytest tests/test_trocr_tokenizer.py (or just pytest).
# Q: Why is no real dataset needed? A: The tokenizer only maps characters to numbers. 20 hand-written Devanagari
#    words with conjuncts, vowel signs and the virama are enough to test it, and no image or model is involved.

# 20 test words with conjuncts, vowel signs and the virama, to check that multi-code-point words survive a round trip.
WORDS = [
    "किताब", "विद्यालय", "क्षत्रिय", "ज्ञान", "प्रेम", "स्वतंत्रता", "धर्म", "राष्ट्र", "श्रीमान", "द्वार",
    "भारत", "हिन्दी", "पुस्तकालय", "कृपया", "उत्तर", "प्रश्न", "सम्मान", "संस्कृत", "विज्ञान", "मित्र",
]


def make():
    """Build a tokenizer whose vocabulary is all distinct (NFC) characters of WORDS, sorted for a fixed order."""
    # Q: What does this helper do? A: It builds a CharTokenizer whose characters are all distinct NFC characters found
    #    in WORDS, sorted. All tests (except the file tests) use it. It is a helper, not a test.
    # Q: Why sorted and why NFC? A: Sorting gives the same id order on every run, so tests are repeatable. NFC makes
    #    the vocabulary use the same code-point form that encode() uses, so lookups match.
    # Q: Why use a set? A: A set removes repeated characters; the tokenizer also drops duplicates, but this keeps
    #    the list clean and the order sorted.
    return CharTokenizer(sorted({c for w in WORDS for c in unicodedata.normalize("NFC", w)}))


def test_special_ids():
    """<pad>, <bos>, <eos>, <unk> must have ids 0, 1, 2, 3 and the first real character must have id 4."""
    # Q: What does this test check? A: The four special tokens have the fixed ids pad=0, bos=1, eos=2, unk=3, the
    #    tokenizer attributes agree with the constants, and the first real character gets id 4.
    # Q: Why does it matter? What bug would make it fail? A: The model code (shift_right, the loss, generation)
    #    hard-codes these ids. If the ids moved, e.g. a character got id 1, the decoder would treat it as <bos>.
    #    It would also fail if the specials were not placed first in the vocabulary.
    # Q: How is the input built and why is it enough? A: make() builds a normal tokenizer. Nothing more is needed,
    #    because the ids are fixed by the class and do not depend on the characters.
    # Q: Which project code does it exercise? A: CharTokenizer.__init__ and the constants PAD, BOS, EOS, UNK.
    tok = make()
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (PAD, BOS, EOS, UNK) == (0, 1, 2, 3)
    assert tok.stoi[tok.chars[0]] == 4


def test_roundtrip_20_words():
    """decode(encode(word)) must give the word back; ids end with <eos> and never contain <bos>."""
    # Q: What does this test check? A: For each of the 20 words, the ids end with <eos>, contain no <bos>, and
    #    decode(encode(word)) gives back the NFC form of the word.
    # Q: Why does it matter? What bug would make it fail? A: A broken round trip means labels are corrupted before
    #    training, or predictions are written wrongly. It would fail if a matra / virama were lost, if <eos> were
    #    missing (the model could not learn to stop), or if <bos> were added in encode (the model adds it itself).
    # Q: How is the input built and why is it enough? A: WORDS has conjuncts (क्ष, ज्ञ, त्र), vowel signs and the
    #    virama, the hardest cases because one visible letter is several code points. If these pass, plain words pass.
    # Q: Which project functions does it exercise? A: CharTokenizer.encode and CharTokenizer.decode.
    tok = make()
    for w in WORDS:
        ids = tok.encode(w)
        assert ids[-1] == EOS and BOS not in ids
        assert tok.decode(ids) == unicodedata.normalize("NFC", w)


def test_unknown_char_and_truncation():
    """A character outside the vocabulary becomes <unk>; long words are cut to max_length and still end with <eos>."""
    # Q: What does this test check? A: The Latin letter "Z" (not in the vocabulary) becomes <unk>, and a word of 100
    #    characters encoded with max_length=8 gives exactly 8 ids, the last one being <eos>.
    # Q: Why does it matter? What bug would make it fail? A: Real data can contain rare characters; they must not
    #    crash training. And the model has a fixed maximum length, so too-long labels must be cut but still end with
    #    <eos>. It would fail if lookup raised an error, or if truncation removed <eos> or gave more than
    #    max_length ids.
    # Q: How is the input built and why is it enough? A: "Z" is certainly outside a Devanagari vocabulary. "क" * 100 is
    #    a known character repeated far beyond the limit, so cutting must happen: 7 characters + <eos> = 8.
    # Q: Which project function does it exercise? A: CharTokenizer.encode (the stoi.get(c, UNK) lookup and the
    #    [: max_length - 1] cut).
    tok = make()
    # "Z" is a Latin letter, so it is not in the Devanagari vocabulary.
    assert tok.encode("Z")[0] == UNK
    # 100 characters cut to 8 ids in total: 7 characters + <eos>.
    assert len(tok.encode("क" * 100, max_length=8)) == 8
    assert tok.encode("क" * 100, max_length=8)[-1] == EOS


def test_batch_encode_pads_with_minus100():
    """batch_encode returns a LongTensor (B, L); the shorter word is padded with -100 and still decodes correctly."""
    # Q: What does this test check? A: batch_encode returns a torch.long tensor with 2 rows; the shorter word's row
    #    contains -100 padding; and decode() of that padded row still gives "धर्म".
    # Q: Why does it matter? What bug would make it fail? A: Labels in a batch must have equal length, and the
    #    padding must be -100 so the loss ignores it. If padding used <pad> (0) the model would learn to predict
    #    padding; if decode did not skip -100, the output text would be wrong or would crash.
    # Q: How is the input built and why is it enough? A: "प्रेम" has 5 code points and "धर्म" has 4, so the encoded
    #    lengths are 6 and 5 (with <eos>). The shorter word needs exactly one padding slot, enough to test padding.
    # Q: Which project functions does it exercise? A: CharTokenizer.batch_encode and CharTokenizer.decode.
    tok = make()
    b = tok.batch_encode(["प्रेम", "धर्म"])
    assert b.dtype == torch.long and b.shape[0] == 2
    # "धर्म" is shorter than "प्रेम", so its row must contain at least one -100.
    assert (b[1] == -100).any()
    # decode() ignores the -100 padding.
    assert tok.decode(b[1]) == "धर्म"


def test_save_load(tmp_path):
    """A tokenizer saved to JSON and loaded again must have the identical id -> token table."""
    # Q: What does this test check? A: After tok.save(path) and CharTokenizer.load(path), the new tokenizer has the
    #    same itos list (same tokens in the same id order) as the original.
    # Q: Why does it matter? What bug would make it fail? A: A trained model only works with the exact id order it was
    #    trained with. If saving lost a character, changed the order, or broke Devanagari (wrong file encoding), the
    #    loaded tokenizer would decode predictions into wrong letters.
    # Q: How is the input built and why is it enough? A: The tokenizer from make() is saved to a JSON file in the
    #    pytest temporary folder (tmp_path), so no real file is touched. Comparing itos covers both content and order.
    # Q: Which project functions does it exercise? A: CharTokenizer.save and CharTokenizer.load.
    tok = make()
    p = tmp_path / "tok.json"
    tok.save(p)
    assert CharTokenizer.load(p).itos == tok.itos


def test_vocab_file(tmp_path):
    """from_vocab_file reads one character per line: 3 characters + 4 specials = 7 tokens."""
    # Q: What does this test check? A: from_vocab_file reads a text file with one character per line; 3 characters plus
    #    4 special tokens give len(tok) == 7, and encode / decode of "कि" works.
    # Q: Why does it matter? What bug would make it fail? A: The real vocab.txt is read this way before training. A
    #    file ends with a newline, and if that made an empty token, the vocabulary size and all ids would be wrong. The
    #    test fails if empty lines are not dropped or if the file is read with a wrong encoding.
    # Q: How is the input built and why is it enough? A: A tiny vocab.txt with "क", "ा", "ि" is written (UTF-8, with a
    #    final newline) in tmp_path. Three characters are enough to count the vocabulary and test a round trip.
    # Q: Which project function does it exercise? A: CharTokenizer.from_vocab_file (plus __len__, encode, decode).
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(["क", "ा", "ि"]) + "\n", encoding="utf-8")
    tok = CharTokenizer.from_vocab_file(p)
    # The file's final newline must not create an extra empty token.
    assert len(tok) == 7 and tok.decode(tok.encode("कि")) == "कि"
