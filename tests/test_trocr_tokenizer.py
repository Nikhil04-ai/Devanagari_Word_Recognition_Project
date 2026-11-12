import unicodedata

import torch

from src.trocr.tokenizer import BOS, EOS, PAD, UNK, CharTokenizer

WORDS = [
    "किताब", "विद्यालय", "क्षत्रिय", "ज्ञान", "प्रेम", "स्वतंत्रता", "धर्म", "राष्ट्र", "श्रीमान", "द्वार",
    "भारत", "हिन्दी", "पुस्तकालय", "कृपया", "उत्तर", "प्रश्न", "सम्मान", "संस्कृत", "विज्ञान", "मित्र",
]


def make():
    return CharTokenizer(sorted({c for w in WORDS for c in unicodedata.normalize("NFC", w)}))


def test_special_ids():
    tok = make()
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (PAD, BOS, EOS, UNK) == (0, 1, 2, 3)
    assert tok.stoi[tok.chars[0]] == 4


def test_roundtrip_20_words():
    tok = make()
    for w in WORDS:
        ids = tok.encode(w)
        assert ids[-1] == EOS and BOS not in ids
        assert tok.decode(ids) == unicodedata.normalize("NFC", w)


def test_unknown_char_and_truncation():
    tok = make()
    assert tok.encode("Z")[0] == UNK
    assert len(tok.encode("क" * 100, max_length=8)) == 8
    assert tok.encode("क" * 100, max_length=8)[-1] == EOS


def test_batch_encode_pads_with_minus100():
    tok = make()
    b = tok.batch_encode(["प्रेम", "धर्म"])
    assert b.dtype == torch.long and b.shape[0] == 2
    assert (b[1] == -100).any()
    assert tok.decode(b[1]) == "धर्म"


def test_save_load(tmp_path):
    tok = make()
    p = tmp_path / "tok.json"
    tok.save(p)
    assert CharTokenizer.load(p).itos == tok.itos


def test_vocab_file(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("\n".join(["क", "ा", "ि"]) + "\n", encoding="utf-8")
    tok = CharTokenizer.from_vocab_file(p)
    assert len(tok) == 7 and tok.decode(tok.encode("कि")) == "कि"
