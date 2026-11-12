"""Character-level tokenizer for the TrOCR model (Contract 2).

Special tokens: <pad>=0, <bos>=1, <eos>=2, <unk>=3; real characters start at 4.
The vocabulary file has one Unicode code point per line (built from train.csv only).
"""
import json
import unicodedata

import torch

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss


class CharTokenizer:
    def __init__(self, chars):
        # keep order, drop duplicates and any accidental special-token strings
        seen, clean = set(), []
        for c in chars:
            if c in seen or c in SPECIALS:
                continue
            seen.add(c)
            clean.append(c)
        self.chars = clean
        self.itos = SPECIALS + self.chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = PAD, BOS, EOS, UNK

    # ---------- construction ----------
    @classmethod
    def from_vocab_file(cls, path):
        """Read vocab.txt: one code point per line (UTF-8). Only '\n' / '\r' are stripped,
        so a space character on its own line would be kept."""
        with open(path, encoding="utf-8") as f:
            chars = [line.rstrip("\r\n") for line in f]
        chars = [c for c in chars if c != ""]
        return cls(chars)

    # ---------- encode / decode ----------
    def encode(self, text, max_length=32):
        """text -> ids ending with <eos>. No <bos> (the model adds decoder_start_token_id).
        Truncated so that the result, including <eos>, has at most max_length ids."""
        text = unicodedata.normalize("NFC", str(text).strip())
        ids = [self.stoi.get(c, UNK) for c in text][: max_length - 1]
        return ids + [EOS]

    def batch_encode(self, texts, max_length=32):
        """List of texts -> LongTensor (B, L) padded with -100 (ready to use as labels)."""
        seqs = [self.encode(t, max_length) for t in texts]
        longest = max(len(s) for s in seqs)
        out = torch.full((len(seqs), longest), IGNORE_INDEX, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def decode(self, ids, skip_special=True):
        """ids -> NFC string. Stops at the first <eos>. Ignores -100 and out-of-range ids."""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        out = []
        for i in ids:
            if i == EOS:
                break
            if i < 0 or i >= len(self.itos):
                continue
            if skip_special and i < len(SPECIALS):
                continue
            out.append(self.itos[i])
        return unicodedata.normalize("NFC", "".join(out))

    def __len__(self):
        return len(self.itos)

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])

    def to_dict(self):
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        return cls(d["chars"])


if __name__ == "__main__":
    words = ["किताब", "विद्यालय", "क्षत्रिय", "ज्ञान", "प्रेम", "स्वतंत्रता", "धर्म", "राष्ट्र", "श्रीमान", "द्वार"]
    tok = CharTokenizer(sorted({c for w in words for c in w}))
    for w in words:
        assert tok.decode(tok.encode(w)) == w
    print("vocab size", len(tok), "| example ids:", tok.encode("विद्यालय"))
