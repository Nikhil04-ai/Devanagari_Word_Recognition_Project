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

# 20 test words with conjuncts, vowel signs and the virama, to check that multi-code-point words survive a round trip.
WORDS = [
    "किताब", "विद्यालय", "क्षत्रिय", "ज्ञान", "प्रेम", "स्वतंत्रता", "धर्म", "राष्ट्र", "श्रीमान", "द्वार",
    "भारत", "हिन्दी", "पुस्तकालय", "कृपया", "उत्तर", "प्रश्न", "सम्मान", "संस्कृत", "विज्ञान", "मित्र",
]


def make():
    """Build a tokenizer whose vocabulary is all distinct (NFC) characters of WORDS, sorted for a fixed order."""
    return CharTokenizer(sorted({c for w in WORDS for c in unicodedata.normalize("NFC", w)}))


def test_special_ids():
    """<pad>, <bos>, <eos>, <unk> must have ids 0, 1, 2, 3 and the first real character must have id 4."""
    tok = make()
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (PAD, BOS, EOS, UNK) == (0, 1, 2, 3)
    assert tok.stoi[tok.chars[0]] == 4


def test_roundtrip_20_words():
    """decode(encode(word)) must give the word back; ids end with <eos> and never contain <bos>."""
    tok = make()
    for w in WORDS:
        ids = tok.encode(w)
        assert ids[-1] == EOS and BOS not in ids
        assert tok.decode(ids) == unicodedata.normalize("NFC", w)


def test_unknown_char_and_truncation():
    """A character outside the vocabulary becomes <unk>; long words are cut to max_length and still end with <eos>."""
    tok = make()
    # "Z" is a Latin letter, so it is not in the Devanagari vocabulary.
    assert tok.encode("Z")[0] == UNK
    # 100 characters cut to 8 ids in total: 7 characters + <eos>.
    assert len(tok.encode("क" * 100, max_length=8)) == 8
    assert tok.encode("क" * 100, max_length=8)[-1] == EOS


def test_batch_encode_pads_with_minus100():
    """batch_encode returns a LongTensor (B, L); the shorter word is padded with -100 and still decodes correctly."""
    tok = make()
    b = tok.batch_encode(["प्रेम", "धर्म"])
    assert b.dtype == torch.long and b.shape[0] == 2
    # "धर्म" is shorter than "प्रेम", so its row must contain at least one -100.
    assert (b[1] == -100).any()
    # decode() ignores the -100 padding.
    assert tok.decode(b[1]) == "धर्म"


def test_save_load(tmp_path):
    """A tokenizer saved to JSON and loaded again must have the identical id -> token table."""
    tok = make()
    p = tmp_path / "tok.json"
    tok.save(p)
    assert CharTokenizer.load(p).itos == tok.itos


def test_vocab_file(tmp_path):
    """from_vocab_file reads one character per line: 3 characters + 4 specials = 7 tokens."""
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(["क", "ा", "ि"]) + "\n", encoding="utf-8")
    tok = CharTokenizer.from_vocab_file(p)
    # The file's final newline must not create an extra empty token.
    assert len(tok) == 7 and tok.decode(tok.encode("कि")) == "कि"
