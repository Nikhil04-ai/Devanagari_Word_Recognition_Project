"""Model C: TrOCR-style Transformer OCR with a character-level Devanagari vocabulary.

Approach 1 ("finetune")       : pre-trained TrOCR-small encoder + pre-trained decoder whose token
                                embeddings / output layer are replaced by OUR character vocabulary.
Approach 2 ("custom_decoder") : pre-trained TrOCR-small encoder + a new nn.TransformerDecoder
                                (4 layers, d_model = encoder hidden size, 6 heads) trained from scratch.

Both expose the same two calls used by train.py / evaluate.py / predict.py:
    out = model(pixel_values=..., labels=...)          -> out.loss, out.logits
    ids = model.generate(pixel_values=..., num_beams=, max_length=)   -> LongTensor (B, L), starts with <bos>

Where it fits in the pipeline: dataset.py gives image tensors and label ids; this file turns an
image into a sequence of character ids; train.py and evaluate.py use it.

How the model works (encoder-decoder):
    1. ENCODER (a Vision Transformer): cuts the image into patches and produces one feature vector
       per patch. Output shape (B, N, d): B images, N patch positions, d = encoder hidden size.
    2. DECODER (a Transformer): writes the word one character at a time. At each step it looks at
       the characters written so far (self-attention) and at the image features (cross-attention),
       and gives a score for every character in the vocabulary (logits, shape (B, L, vocab)).
    Why the pre-trained encoder is reused in both approaches: it has already learned to read
    handwriting-like images, so we do not start from zero.
"""
# types: SimpleNamespace is used to return an object with .loss and .logits (same style as Hugging Face outputs).
import types

# torch: tensors. nn: layers. F: functional ops such as cross_entropy and log_softmax.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Special token ids and the label padding value (see tokenizer.py).
from .tokenizer import BOS, EOS, IGNORE_INDEX, PAD

MAX_POS = 34  # learned positional embeddings of the custom decoder (>= max_length + 2)
# (max_length = 32 labels incl. <eos>, plus the <bos> in front, gives 33; 34 leaves room to spare.)


# ----------------------------------------------------------------------------------------------
# Approach 2: custom decoder on top of the pre-trained encoder
# ----------------------------------------------------------------------------------------------
def shift_right(labels):
    """labels (…, -100 padded, ends with <eos>) -> decoder inputs: [<bos>, c1, …, cn], pad = <pad>.

    Input: LongTensor (B, L) of label ids. Returns: LongTensor (B, L), same shape.
    Why: this is teacher forcing. At step t the decoder receives the TRUE previous character
    (not its own guess) and must predict character t. So the input is the labels moved one place
    to the right, with <bos> placed at the start. The last label (<eos>) is dropped from the input.
    Example: labels [k, h, EOS, -100] -> decoder input [BOS, k, h, EOS]."""
    # Start with a tensor of the same shape that is full of <pad>.
    ids = labels.new_full(labels.shape, PAD)
    # First position is always <bos>.
    ids[:, 0] = BOS
    # Copy the labels one place to the right (the last label is not copied).
    ids[:, 1:] = labels[:, :-1]
    # -100 is only valid as a loss label, not as an embedding index, so it is replaced by <pad>.
    return ids.masked_fill(ids == IGNORE_INDEX, PAD)


class CustomDecoderOCR(nn.Module):
    """Pre-trained image encoder + a Transformer decoder that is trained from scratch.

    This is the backup approach (cfg.approach = "custom_decoder"). The decoder here is plain PyTorch,
    so no pre-trained text decoder is used. It has the same forward / generate interface as approach 1."""
    # Flag read by generate_ids(): this model has its own generate(), so no Hugging Face options are passed.
    is_custom = True

    def __init__(self, encoder, vocab_size, num_layers=4, heads=6, ff=1536, dropout=0.1):
        """Inputs: encoder = pre-trained ViT encoder; vocab_size = number of tokens (with specials);
        num_layers / heads / ff / dropout = decoder size (layers, attention heads, feed-forward width, dropout)."""
        super().__init__()
        # d = size of one feature vector; the decoder must use the same size as the encoder output.
        d = encoder.config.hidden_size
        self.encoder = encoder
        self.vocab_size = vocab_size
        # Character embedding: id -> vector of size d. padding_idx=PAD keeps the <pad> vector at zero and gets no gradient.
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        # Learned position embedding: position 0..MAX_POS-1 -> vector of size d (tells the decoder where in the word it is).
        self.pos = nn.Embedding(MAX_POS, d)
        # One decoder layer = masked self-attention + cross-attention to the image + feed-forward.
        # batch_first=True means tensors are (B, L, d). norm_first=True applies LayerNorm before each sub-layer.
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        # Stack of num_layers such layers, followed by a final LayerNorm.
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d))
        # Output layer: vector of size d -> one score (logit) per vocabulary token.
        self.head = nn.Linear(d, vocab_size)
        # Small random start values for the new embeddings.
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            # The init above overwrote the <pad> row, so it is set back to zeros.
            self.embed.weight[PAD].zero_()

    def encode_images(self, pixel_values):
        """pixel_values (B, 3, H, W) -> image features "memory" (B, N, d), one vector per patch (plus any special token)."""
        return self.encoder(pixel_values=pixel_values).last_hidden_state

    def decode(self, dec_in, memory):
        """Run the decoder once.

        Inputs: dec_in LongTensor (B, L) = characters written so far (starting with <bos>);
        memory FloatTensor (B, N, d) = image features. Returns: logits FloatTensor (B, L, vocab).
        Logits at position t are the scores for the NEXT character after dec_in[:, t]."""
        L = dec_in.shape[1]
        # Character vector + position vector. (B, L) -> (B, L, d); pos is (L, d) and [None] makes it (1, L, d) to broadcast over B.
        x = self.embed(dec_in) + self.pos(torch.arange(L, device=dec_in.device))[None]
        # Causal mask (L, L): True above the diagonal = "position t may NOT look at positions after t".
        # Without it the decoder could copy the answer during training.
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dec_in.device), 1)
        # tgt_key_padding_mask (B, L): True on <pad> positions so that attention ignores them.
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=(dec_in == PAD) & (dec_in != BOS))
        # (B, L, d) -> (B, L, vocab)
        return self.head(h)

    def forward(self, pixel_values, labels=None, **_):
        """Training / validation pass. pixel_values (B, 3, H, W), labels (B, L) padded with -100.
        Returns an object with .loss (scalar) and .logits (B, L, vocab), like a Hugging Face model.
        `labels` must be given in practice (it is needed for shift_right and for the loss);
        `**_` swallows extra keyword arguments so that the call is the same as for the Hugging Face model."""
        memory = self.encode_images(pixel_values)
        # Teacher forcing: the decoder input is the labels shifted right.
        logits = self.decode(shift_right(labels), memory)
        # Flatten (B, L, vocab) -> (B*L, vocab) and (B, L) -> (B*L); positions with -100 are skipped by ignore_index.
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX)
        return types.SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, pixel_values, num_beams=1, max_length=32, **_):
        """Write words for a batch of images (no gradients, used for validation and test).

        num_beams=1 -> greedy decoding; num_beams>1 -> beam search (done per image).
        Returns LongTensor (B, L) that starts with <bos>; short rows are padded with <pad>."""
        # The image is encoded only once; every decoding step reuses the same memory.
        memory = self.encode_images(pixel_values)
        if num_beams <= 1:
            return self._greedy(memory, max_length)
        # Beam search handles one image at a time (memory[i:i+1] keeps the batch axis: (1, N, d)).
        outs = [self._beam(memory[i: i + 1], num_beams, max_length) for i in range(memory.shape[0])]
        # The best sequences can have different lengths, so pad them with <pad> into one (B, width) tensor.
        width = max(len(o) for o in outs)
        res = torch.full((len(outs), width), PAD, dtype=torch.long, device=memory.device)
        for i, o in enumerate(outs):
            res[i, : len(o)] = torch.tensor(o, device=memory.device)
        return res

    def _greedy(self, memory, max_length):
        """Greedy decoding: at every step take the single most likely next character.
        Input: memory (B, N, d). Returns LongTensor (B, L), L <= max_length + 1, starts with <bos>.
        Fast and simple, but it can not take back an early wrong choice (beam search can)."""
        B = memory.shape[0]
        # Every sequence starts with <bos>: shape (B, 1).
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=memory.device)
        # done[b] becomes True once sequence b has produced <eos>.
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        # MAX_POS - 1 steps at most, so positions never go past the position table.
        for _ in range(min(max_length, MAX_POS - 1)):
            # Run the decoder on everything written so far; take logits of the last position (B, vocab) and argmax -> (B,).
            nxt = self.decode(ids, memory)[:, -1].argmax(-1)
            # Finished sequences get <pad> instead of new characters.
            nxt = nxt.masked_fill(done, PAD)
            # Append the new character: (B, L) -> (B, L+1).
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == EOS
            # Stop early when every sequence has finished.
            if done.all():
                break
        return ids

    def _beam(self, memory, k, max_length):
        """Beam search for one image; returns the best id list (starting with <bos>).

        Idea: keep the k best partial words (the "beams") instead of only one. At each step every
        beam is extended with its k most likely next characters, and only the k best extended
        sequences are kept. Sequences that end with <eos> are moved to `finished` with a
        length-normalised score (sum of log-probabilities / number of generated tokens), so
        short words are not unfairly preferred.
        Input: memory (1, N, d); k = beam width. Returns: list of ints."""
        dev = memory.device
        # Each beam = (id list, sum of log-probabilities). Start with one beam that holds only <bos>.
        beams = [([BOS], 0.0)]
        finished = []
        for _ in range(min(max_length, MAX_POS - 1)):
            # All beams have the same length here -> tensor (num_beams, L).
            seqs = torch.tensor([b[0] for b in beams], device=dev)
            # Repeat the image features for each beam (expand makes a view, no copy): (1,N,d) -> (num_beams,N,d).
            # Take the last position and turn scores into log-probabilities: (num_beams, vocab).
            logp = F.log_softmax(self.decode(seqs, memory.expand(len(beams), -1, -1))[:, -1], -1)
            cand = []
            for bi, (seq, score) in enumerate(beams):
                # The k best next characters for this beam.
                top = logp[bi].topk(k)
                # New candidate = old sequence + one character, score = old score + log-probability of that character.
                cand += [(seq + [t.item()], score + v.item()) for v, t in zip(top.values, top.indices)]
            # Best candidates first.
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in cand:
                if seq[-1] == EOS:
                    finished.append((seq, score / (len(seq) - 1)))  # length-normalised
                else:
                    beams.append((seq, score))
                # Keep at most k unfinished beams.
                if len(beams) == k:
                    break
            # Stop when no beam is left, or when we have k finished words and the best finished
            # score is already better than the best running beam's per-token score.
            if not beams or (len(finished) >= k and max(f[1] for f in finished) > beams[0][1] / len(beams[0][0])):
                break
        # If nothing finished within max_length, fall back to the running beams.
        if not finished:
            finished = [(s, sc / max(len(s) - 1, 1)) for s, sc in beams]
        # Return the sequence with the best normalised score.
        return max(finished, key=lambda f: f[1])[0]


# ----------------------------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------------------------
def build_model(cfg, tokenizer, load_pretrained=True):
    """Create the model for cfg.approach. load_pretrained=False builds the same architecture with
    random weights (used when restoring one of our own checkpoints).

    Inputs: cfg (config namespace), tokenizer (its length is the vocabulary size).
    Returns: an nn.Module that can be called as model(pixel_values=..., labels=...) and model.generate(...)."""
    # Imported here so that the rest of this file (shift_right, CustomDecoderOCR) works without Hugging Face loading.
    from transformers import AutoModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    # n = vocabulary size of OUR tokenizer (4 specials + characters).
    n = len(tokenizer)
    if load_pretrained:
        # Download / load the pre-trained TrOCR (encoder + decoder with English weights).
        base = VisionEncoderDecoderModel.from_pretrained(cfg.pretrained)
    else:
        # Same architecture from the config only, with random weights. Used by evaluate.py,
        # which then loads our own trained weights, so the big pre-trained download is not needed.
        base = VisionEncoderDecoderModel(VisionEncoderDecoderConfig.from_pretrained(cfg.pretrained))

    if cfg.approach == "custom_decoder":
        # Approach 2: keep only the pre-trained encoder and attach our new decoder.
        return CustomDecoderOCR(base.encoder, n)
    if cfg.approach != "finetune":
        raise ValueError(f"unknown approach {cfg.approach!r} (use 'finetune' or 'custom_decoder')")

    # ---- Approach 1: replace the English sub-word vocabulary by our character vocabulary ----
    # Why: the pre-trained decoder's embedding table and output layer are made for the English
    # token ids. Our ids mean different things (characters, not English sub-words), so those two
    # parts must be new. The other decoder layers (attention, feed-forward) keep their pre-trained weights.
    dec = base.decoder
    # Resize the embedding table to n rows (the output layer is resized with it when they are tied).
    dec.resize_token_embeddings(n)
    emb = dec.get_input_embeddings()
    emb.padding_idx = PAD  # the English model used padding_idx=1, which is <bos> for us
    # The old rows hold English token vectors, which are meaningless here, so start from small random values.
    nn.init.normal_(emb.weight, std=0.02)
    with torch.no_grad():
        # <pad> vector must stay zero.
        emb.weight[PAD].zero_()
    out = dec.get_output_embeddings()
    if out.out_features != n:  # not tied / not resized: replace the output layer
        # New linear layer: hidden size -> n scores, without bias.
        head = nn.Linear(out.in_features, n, bias=False)
        dec.set_output_embeddings(head)
        out = head
    # If the output weight is a separate tensor (not shared with the embedding), it also needs a fresh random start.
    # data_ptr() is the memory address, so equal addresses mean the two layers share one weight.
    if out.weight.data_ptr() != emb.weight.data_ptr():
        nn.init.normal_(out.weight, std=0.02)

    # Tell the model (and its decoder config) which ids are the special tokens.
    # decoder_start_token_id = <bos>: the Hugging Face model puts it in front of the labels
    # (the same shift-right idea as shift_right above) and replaces -100 by <pad> when it builds the decoder input.
    for c in (base.config, base.config.decoder):
        c.decoder_start_token_id, c.pad_token_id, c.eos_token_id, c.bos_token_id = BOS, PAD, EOS, BOS
    base.config.vocab_size = n
    base.config.decoder.vocab_size = n
    # Defaults used by model.generate(): the same special tokens, length limit and beams from the config.
    # early_stopping=True stops beam search once enough finished candidates exist; use_cache=True reuses
    # earlier decoder computations so that each new step is cheaper.
    g = base.generation_config
    g.decoder_start_token_id, g.pad_token_id, g.eos_token_id, g.bos_token_id = BOS, PAD, EOS, BOS
    g.max_length, g.num_beams, g.early_stopping, g.use_cache = cfg.max_length, cfg.num_beams, True, True
    return base


def count_params(model, trainable_only=True):
    """Number of parameters. trainable_only=True counts only parameters with requires_grad=True
    (so a frozen encoder is not counted); False counts all of them."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def set_encoder_trainable(model, trainable):
    """Freeze (trainable=False) or unfreeze (True) the image encoder.
    Used with cfg.freeze_encoder_epochs: the first epochs can train only the decoder."""
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def new_param_names(model, cfg):
    """Names of parameters that are NEW (randomly initialised) and so get the larger lr_new.

    Returns a set of names from model.named_parameters(). train.make_optimizer uses it to split
    the parameters into two groups with two learning rates."""
    if cfg.approach == "custom_decoder":
        # Everything except the pre-trained encoder is new.
        return {n for n, _ in model.named_parameters() if not n.startswith("encoder.")}
    # finetune: only the replaced character embeddings and the output layer are new.
    return {n for n, _ in model.named_parameters()
            if "decoder.model.decoder.embed_tokens" in n or "decoder.output_projection" in n}


@torch.no_grad()
def generate_ids(model, pixel_values, num_beams=1, max_length=32):
    """Unified generation for both approaches. Returns ids starting with <bos>.

    Input: pixel_values (B, 3, H, W). Returns LongTensor (B, L). num_beams=1 is greedy, >1 is beam search.
    Why one helper: train.py and evaluate.py then do not need to know which approach is used."""
    kw = dict(pixel_values=pixel_values, num_beams=num_beams, max_length=max_length)
    # The custom model has its own generate() and needs only the three arguments above.
    if not getattr(model, "is_custom", False):
        # The Hugging Face model gets our special-token ids explicitly (the defaults are for English).
        kw.update(decoder_start_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=BOS, use_cache=True)
        if num_beams > 1:
            kw["early_stopping"] = True
    return model.generate(**kw)


# Quick self-check: run `python -m src.trocr.model`. It builds both approaches, runs one forward
# pass on random images and prints the parameter count and the loss.
if __name__ == "__main__":
    from .config import load_config
    from .tokenizer import CharTokenizer

    tok = CharTokenizer(list("कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंँ्"))
    for approach in ("finetune", "custom_decoder"):
        cfg = load_config(None, {"approach": approach})
        model = build_model(cfg, tok)
        # Random "images": batch of 2, 3 channels, 384x384.
        x = torch.randn(2, 3, 384, 384)
        labels = tok.batch_encode(["प्रेम", "धर्म"])
        out = model(pixel_values=x, labels=labels)
        # With random weights the model is about equally unsure about every class, so the starting
        # loss should be near ln(number of classes).
        print(f"{approach}: params={count_params(model) / 1e6:.1f}M  loss={out.loss.item():.3f} "
              f"(~ ln({len(tok)}) = {torch.log(torch.tensor(float(len(tok)))):.2f} at init)")
