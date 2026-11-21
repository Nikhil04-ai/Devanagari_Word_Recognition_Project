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

# ======================================= CONCEPTS USED IN THIS FILE =======================================
# Tokenizer: converts text to integer ids and back. The model only works with numbers. This file is the
#     class CharTokenizer (encode, decode).
# Character-level vocabulary: one token per Unicode code point (letter, matra, virama...), not per word.
#     Built from vocab.txt in from_vocab_file; its size is len(tokenizer).
# Unicode code point: one numbered character. A Devanagari "letter" on screen can be several code points
#     (for example a base letter + virama + another letter). The model predicts code points.
# Matra / virama: matra = vowel sign written with a consonant; virama = mark that joins consonants into a
#     conjunct (क + ् + ष = क्ष). Both are ordinary tokens here.
# NFC normalisation: Unicode rule that gives the same visible text one fixed code point sequence. Used in
#     encode and decode so labels and predictions compare fairly.
# Special tokens: <pad>=0 (filler), <bos>=1 (start, first decoder input), <eos>=2 (end, tells the model to stop),
#     <unk>=3 (character not in the vocabulary). Defined by PAD, BOS, EOS, UNK and SPECIALS.
# Label masking with -100: IGNORE_INDEX; label slots with -100 are skipped by the cross-entropy loss, so padding
#     is not learned. Used in batch_encode and skipped in decode.
# Teacher forcing: while training, the decoder is fed the true previous characters. The tokenizer therefore
#     gives labels without <bos>; the model adds it (see model.shift_right).
# Truncation (max_length): words are cut so that the ids, including <eos>, are at most max_length long.
# Tensor / LongTensor (B, L): a PyTorch array of 64-bit integers; B = batch size, L = longest label.
# ==========================================================================================================

# ======================================= FILE-LEVEL Q&A (VIVA) ============================================
# Q: What part of the project is this file?
# A: The text side of the TrOCR model: it maps Devanagari characters to ids (labels) and the model's predicted
#    ids back to text. dataset.py, train.py and evaluate.py all use it.
# Q: How is it run, and what does it read and write?
# A: It is imported. It reads vocab.txt (one character per line, from the training split only) or a JSON /
#    dict saved in a checkpoint, and it can write that JSON. Running `python -m src.trocr.tokenizer` runs a
#    small round-trip self-check on ten words.
# Q: Why a character vocabulary and not whole words?
# A: It is small and it can write words never seen in training. A word-level vocabulary would be huge and
#    could not output a new word.
# ==========================================================================================================

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
    # Q: What does it do and why does it exist?
    # A: It holds the character vocabulary and converts text to ids and ids to text. One object keeps the
    #    tables (itos, stoi) together with encode, decode, save and load.
    # Q: Why are the special tokens put first, with fixed ids?
    # A: PAD, BOS, EOS, UNK are always 0..3, so the other files (model.py, train.py) can use these constants
    #    without asking the tokenizer. Real characters start at id 4.

    def __init__(self, chars):
        """Build the vocabulary from an iterable of single characters (order is kept)."""
        # Q: What does it do and why does it exist?
        # A: It removes repeated characters and special-token strings, then builds itos (id -> token) and
        #    stoi (token -> id). Every tokenizer is created through this constructor.
        # Q: Why keep the given order and drop duplicates?
        # A: The id of a character is its position in the list. A saved checkpoint must give the same ids again,
        #    and a duplicate would give one character two ids.
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
        # Q: What does it do and why does it exist?
        # A: It reads vocab.txt and returns a CharTokenizer. get_loaders in dataset.py uses it to build the
        #    tokenizer at the start of a run.
        # Q: Why strip only "\r" and "\n", not all whitespace?
        # A: A space character can be a real vocabulary entry on its own line; strip() would turn it into an empty
        #    string and it would be lost. Empty lines are dropped afterwards.
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
        # Q: What does it do and why does it exist?
        # A: It turns one word into a list of character ids that ends with <eos>. These ids are the training
        #    labels in dataset.py.
        # Q: Why truncate to max_length - 1 characters and map unknown characters to UNK?
        # A: One place must stay free for <eos>, so the total never passes max_length. UNK lets the code run
        #    on a character that is not in the vocabulary instead of crashing.
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
        # Q: What does it do and why does it exist?
        # A: It encodes several words and stacks them into one padded (B, L) tensor of labels. It is used by
        #    the self-check in model.py; dataset.py does the same job in collate_fn.
        # Q: Why pad with -100 and not with <pad>?
        # A: The cross-entropy loss skips -100, so the model is not trained to predict padding. With <pad>
        #    as the label it would be punished for not predicting <pad> after the word.
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
        # Q: What does it do and why does it exist?
        # A: It turns the ids predicted by the model (or a label row) back into a Devanagari string. train.py
        #    and evaluate.py use it to get the predicted words that are scored with CER and WER.
        # Q: Why stop at <eos> and skip ids below 0 or out of range?
        # A: <eos> means the word is finished, so what follows is not part of the answer. -100 (label padding)
        #    or a bad id has no character, so it is skipped instead of raising an IndexError.
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
        # Q: What does it do and why does it exist?
        # A: It makes len(tokenizer) work and returns the vocabulary size (4 specials + characters).
        #    model.build_model uses it as the number of output classes.
        # Q: Why does the size include the special tokens?
        # A: The model can predict <pad>, <bos>, <eos> and <unk> too, so the output layer needs one score
        #    for each of them as well as for every character.
        return len(self.itos)

    # ---------- persistence ----------
    def save(self, path):
        """Write the character list to a JSON file. ensure_ascii=False keeps Devanagari readable in the file."""
        # Q: What does it do and why does it exist?
        # A: It writes the character list to a JSON file, so a vocabulary can be stored and loaded again
        #    with the same id order.
        # Q: Why ensure_ascii=False and utf-8?
        # A: Otherwise json would write every Devanagari letter as an escape like क. With these settings
        #    the file shows the real characters and is easy to read and check.
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        """Read a JSON file written by `save` and return the tokenizer."""
        # Q: What does it do and why does it exist?
        # A: It is the reverse of save: it reads the "chars" list from the JSON file and builds a tokenizer.
        # Q: Why is it a classmethod?
        # A: It creates a new object, so it needs the class (cls) and no existing instance. The result is the
        #    same as the tokenizer that was saved, because the character order gives the ids.
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])

    def to_dict(self):
        """Tokenizer -> small dict. train.py stores it inside the checkpoint, so a checkpoint
        always carries the exact vocabulary (and id order) it was trained with."""
        # Q: What does it do and why does it exist?
        # A: It returns the character list as a small dict, which can be stored inside a checkpoint file.
        # Q: Why store the characters and not the special tokens?
        # A: The specials are always the same (ids 0..3) and are added by __init__, so the real characters in
        #    order are all that is needed to rebuild the same id table.
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        """Rebuild the tokenizer from the dict made by `to_dict` (used by evaluate.py)."""
        # Q: What does it do and why does it exist?
        # A: It is the reverse of to_dict. evaluate.load_checkpoint calls it to get the tokenizer that was
        #    saved in the checkpoint.
        # Q: Why not read vocab.txt again here?
        # A: vocab.txt could have changed since training. The dict in the checkpoint always matches the
        #    weights, so the ids agree with the model's output layer.
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
