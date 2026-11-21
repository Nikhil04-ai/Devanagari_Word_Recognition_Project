"""Character-level tokenizer for the TrOCR model (Contract 2).

Special tokens: <pad>=0, <bos>=1, <eos>=2, <unk>=3; real characters start at 4.
The vocabulary file has one Unicode code point per line (built from train.csv only).

Where it fits in the pipeline: the model cannot read text, it only works with numbers. This file
turns a word label such as "प्रेम" into a list of integer ids (encode) for training, and turns the
ids that the model predicts back into text (decode). It is used by dataset.py (labels), train.py
and evaluate.py (decode predictions), and it is saved inside every checkpoint.

Why one token per Unicode code point: Devanagari words are built from base letters, vowel signs
(matras) and the virama, and each of these is its own code point. A character vocabulary is small
and it can write any word made of known characters, also words never seen in training.

What the special tokens are for:
    <pad> = filler used to make sequences the same length inside a batch.
    <bos> = "beginning of sequence"; the first input given to the decoder.
    <eos> = "end of sequence"; the model predicts it when the word is finished.
    <unk> = "unknown"; used for a character that is not in the vocabulary.
"""
# json: save / load the character list. unicodedata: Unicode normalisation (NFC).
import json
import unicodedata

# torch: the label batches are returned as tensors.
import torch

# Fixed ids of the special tokens. They always come first, so real characters start at id 4.
PAD, BOS, EOS, UNK = 0, 1, 2, 3
# Text form of the special tokens, in id order (index in this list == id).
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss
# (-100 is the default `ignore_index` of PyTorch's cross-entropy, so padded label slots add no loss.)


class CharTokenizer:
    """Maps characters <-> integer ids.

    Attributes: `chars` (list of real characters), `itos` (id -> token string, specials first),
    `stoi` (token string -> id), and the ids `pad_id`, `bos_id`, `eos_id`, `unk_id`.
    Why a class: the vocabulary has to be kept together with encode / decode / save / load."""

    def __init__(self, chars):
        """Build the vocabulary from an iterable of single characters (order is kept)."""
        # keep order, drop duplicates and any accidental special-token strings
        seen, clean = set(), []
        for c in chars:
            # Skip repeated characters and strings like "<pad>" so that ids stay unique.
            if c in seen or c in SPECIALS:
                continue
            seen.add(c)
            clean.append(c)
        self.chars = clean
        # id -> token. Specials take ids 0..3, then the characters follow in the given order.
        self.itos = SPECIALS + self.chars
        # token -> id (the reverse table, for fast look-up while encoding).
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = PAD, BOS, EOS, UNK

    # ---------- construction ----------
    @classmethod
    def from_vocab_file(cls, path):
        """Read vocab.txt: one code point per line (UTF-8). Only '\n' / '\r' are stripped,
        so a space character on its own line would be kept.

        Input: path of vocab.txt. Returns: a CharTokenizer.
        Why: vocab.txt is written once from the training split only, so validation and test
        words cannot leak into the vocabulary."""
        with open(path, encoding="utf-8") as f:
            # rstrip("\r\n") removes only the line ending, not other whitespace.
            chars = [line.rstrip("\r\n") for line in f]
        # Drop empty lines (for example a blank line at the end of the file).
        chars = [c for c in chars if c != ""]
        return cls(chars)

    # ---------- encode / decode ----------
    def encode(self, text, max_length=32):
        """text -> ids ending with <eos>. No <bos> (the model adds decoder_start_token_id).
        Truncated so that the result, including <eos>, has at most max_length ids.

        Input: one word (str). Returns: list of ints, length 1..max_length, last item is EOS.
        Why <eos> at the end: the model must learn when to stop writing.
        Why no <bos>: the model shifts the labels right and puts <bos> in front by itself (teacher forcing)."""
        # NFC makes the same visible word always have the same code points, so labels are consistent.
        text = unicodedata.normalize("NFC", str(text).strip())
        # Look up every character; a character not in the vocabulary becomes UNK.
        # [: max_length - 1] keeps one free place for <eos>.
        ids = [self.stoi.get(c, UNK) for c in text][: max_length - 1]
        return ids + [EOS]

    def batch_encode(self, texts, max_length=32):
        """List of texts -> LongTensor (B, L) padded with -100 (ready to use as labels).

        Input: list of B words. Returns: LongTensor of shape (B, L), where L is the length of the
        longest encoded word in this batch (not always max_length).
        Why -100 and not <pad>: the loss ignores label slots that hold -100, so padding is not learned."""
        seqs = [self.encode(t, max_length) for t in texts]
        longest = max(len(s) for s in seqs)
        # Start with a (B, L) tensor full of -100, then copy each real sequence into its row.
        out = torch.full((len(seqs), longest), IGNORE_INDEX, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def decode(self, ids, skip_special=True):
        """ids -> NFC string. Stops at the first <eos>. Ignores -100 and out-of-range ids.

        Input: list or 1-D tensor of ids (one word). Returns: str.
        Why stop at <eos>: anything after it is padding or garbage from the model."""
        # Accept a tensor as well as a list.
        if torch.is_tensor(ids):
            ids = ids.tolist()
        out = []
        for i in ids:
            if i == EOS:
                break
            # Skip -100 (label padding) and ids that are not in the vocabulary.
            if i < 0 or i >= len(self.itos):
                continue
            # Skip <pad>, <bos>, <unk> (ids 0..3) unless the caller wants to see them.
            if skip_special and i < len(SPECIALS):
                continue
            out.append(self.itos[i])
        return unicodedata.normalize("NFC", "".join(out))

    def __len__(self):
        """Vocabulary size including the 4 special tokens. This is the number of output classes of the model."""
        return len(self.itos)

    # ---------- persistence ----------
    def save(self, path):
        """Write the character list to a JSON file. ensure_ascii=False keeps Devanagari readable in the file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        """Read a JSON file written by `save` and return the tokenizer."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])

    def to_dict(self):
        """Tokenizer -> small dict. train.py stores it inside the checkpoint, so a checkpoint
        always carries the exact vocabulary (and id order) it was trained with."""
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        """Rebuild the tokenizer from the dict made by `to_dict` (used by evaluate.py)."""
        return cls(d["chars"])


# Quick self-check: run `python -m src.trocr.tokenizer`.
if __name__ == "__main__":
    # Ten test words that include conjuncts (क्ष, ज्ञ, त्र) and vowel signs.
    words = ["किताब", "विद्यालय", "क्षत्रिय", "ज्ञान", "प्रेम", "स्वतंत्रता", "धर्म", "राष्ट्र", "श्रीमान", "द्वार"]
    # Vocabulary = all distinct characters of these words, sorted for a stable order.
    tok = CharTokenizer(sorted({c for w in words for c in w}))
    for w in words:
        # Round trip: decode(encode(w)) must give back the same word.
        assert tok.decode(tok.encode(w)) == w
    print("vocab size", len(tok), "| example ids:", tok.encode("विद्यालय"))
