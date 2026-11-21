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

# ======================================= CONCEPTS USED IN THIS FILE =======================================
# TrOCR: Transformer-based OCR from Microsoft: an image encoder plus a text decoder. We load the small
#     handwritten version (cfg.pretrained) and replace its English vocabulary with our characters.
# Encoder-decoder: the encoder reads the image into features, the decoder writes the text using them.
#     In build_model (Hugging Face VisionEncoderDecoderModel) and in CustomDecoderOCR.
# Vision Transformer (ViT): an image model that cuts the image into patches and processes them with Transformer
#     layers. It is the encoder (base.encoder); encode_images returns its last_hidden_state.
# Patches: small square pieces of the image; the encoder gives one feature vector per patch. N in (B, N, d).
# Hidden size d: length of one feature vector (encoder.config.hidden_size). The decoder must use the same d.
# Memory: name for the encoder output that the decoder looks at (variable `memory`).
# Transformer decoder layer: masked self-attention + cross-attention + feed-forward (nn.TransformerDecoderLayer).
# Self-attention: each character position looks at the earlier characters of the word.
# Cross-attention: each character position looks at the image features (memory) to decide what to write.
# Causal mask: hides future positions, so position t cannot see characters after t (torch.triu in decode).
#     Without it the decoder could copy the answer during training.
# Key padding mask (tgt_key_padding_mask): marks <pad> positions that attention must ignore.
# Embedding: a table that turns a token id into a learned vector (self.embed). Positional embedding (self.pos):
#     a learned vector per position 0..MAX_POS-1, because attention itself does not know the order.
# padding_idx=PAD: the <pad> embedding row stays zero and gets no gradient.
# LayerNorm / norm_first / batch_first / dropout / feed-forward (ff) / heads: standard Transformer parts.
#     norm_first=True = normalise before each sub-layer; batch_first=True = tensors are (B, L, d); dropout
#     randomly switches off units while training; heads = parallel attention heads; ff = width of the MLP.
# Logits: raw scores, one per vocabulary token, shape (B, L, vocab). Output layer = self.head (nn.Linear).
# Cross-entropy loss and ignore_index=-100: the loss that compares the logits with the true next character; label
#     positions equal to -100 (padding) add nothing. Used in CustomDecoderOCR.forward.
# Special tokens pad/bos/eos/unk: ids 0/1/2/3 from tokenizer.py. <bos> starts decoding, <eos> ends it.
# Teacher forcing / shift right: while training the decoder gets the TRUE previous characters, so the input is
#     the labels moved one place right with <bos> in front (shift_right).
# Greedy generation: at each step take the most likely character (_greedy); it cannot undo an early mistake.
# Beam search: keep the k best partial words at each step (_beam); scores are summed log-probabilities divided by
#     length (length normalisation) so short words are not favoured. early_stopping stops beam search early.
# Log-softmax: turns logits into log-probabilities; adding them gives the log-probability of a whole word.
# Pre-trained weights and fine-tuning: start from weights already trained on handwriting images and continue
#     training them on our data. New random layers learn from scratch (hence two learning rates in train.py).
# Tied weights: the input embedding and the output layer can share one weight matrix (checked with data_ptr).
# resize_token_embeddings: changes the vocabulary size of the embedding table (to our character count).
# Freezing: requires_grad=False stops a part (the encoder) from being updated (set_encoder_trainable).
# decoder_start_token_id / generation_config: the id the Hugging Face decoder starts with (<bos> here) and the
#     default settings of model.generate (max_length, num_beams, use_cache).
# use_cache: reuse earlier decoder computations so each new generated token is cheaper.
# @torch.no_grad(): switches off gradient tracking, saving memory and time when only predicting.
# Initial loss ~ ln(vocab): a random model is equally unsure about all classes, so cross-entropy is about ln(V).
# ==========================================================================================================

# ======================================= FILE-LEVEL Q&A (VIVA) ============================================
# Q: What part of the project is this file?
# A: Model C, the TrOCR-style network: image in, sequence of character ids out. dataset.py feeds it; train.py
#    trains it; evaluate.py tests it. It offers two approaches ("finetune" and "custom_decoder").
# Q: How is it run, and what does it read and write?
# A: It is imported. `python -m src.trocr.model` builds both approaches, runs one forward pass on random images
#    and prints the parameter count and the initial loss. build_model downloads the pre-trained TrOCR weights
#    from Hugging Face (or only its config when load_pretrained=False). It writes nothing.
# Q: Why do both approaches share the same forward and generate calls?
# A: train.py and evaluate.py can then work with either model without knowing which one it is (generate_ids
#    hides the small differences).
# ==========================================================================================================

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
    # Q: What does it do and why does it exist?
    # A: It builds the decoder input for teacher forcing: <bos> first, then the labels moved one place right.
    #    CustomDecoderOCR.forward calls it so that at step t the decoder sees the true characters before t.
    # Q: Why drop the last label and replace -100 by <pad>?
    # A: The input has the same length as the labels, so the last one (usually <eos>) is not needed as input.
    #    -100 is only valid as a loss label; an embedding lookup with -100 would fail, so <pad> is used.
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
    # Q: What does it do and why does it exist?
    # A: It is the backup model: the pre-trained image encoder plus a small Transformer decoder (embedding,
    #    position embedding, decoder layers, output layer) trained from scratch, with our own generate().
    # Q: Why is it a class with the same forward and generate as the Hugging Face model?
    # A: train.py and evaluate.py then use it exactly like approach 1. It is an nn.Module so that its weights,
    #    state_dict and parameters work with the optimizer and checkpoints.
    # Flag read by generate_ids(): this model has its own generate(), so no Hugging Face options are passed.
    is_custom = True

    def __init__(self, encoder, vocab_size, num_layers=4, heads=6, ff=1536, dropout=0.1):
        """Inputs: encoder = pre-trained ViT encoder; vocab_size = number of tokens (with specials);
        num_layers / heads / ff / dropout = decoder size (layers, attention heads, feed-forward width, dropout)."""
        # Q: What does it do and why does it exist?
        # A: It keeps the encoder and creates the new decoder parts: character embedding, position embedding,
        #    a stack of decoder layers with a final LayerNorm, and the output layer (d -> vocab scores).
        # Q: Why d = encoder hidden size, and why is the <pad> row set back to zero?
        # A: Cross-attention reads the encoder features, so decoder vectors must have the same size d. The
        #    normal init overwrites every row, so the <pad> row is zeroed again to keep padding neutral.
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
        # Q: What does it do and why does it exist?
        # A: It runs the pre-trained Vision Transformer on the images and returns its last hidden states, the
        #    "memory" that the decoder attends to.
        # Q: Why a separate method?
        # A: forward and generate both need the image features. Generation calls it only once per batch, and
        #    every decoding step then reuses the same memory.
        return self.encoder(pixel_values=pixel_values).last_hidden_state

    def decode(self, dec_in, memory):
        """Run the decoder once.

        Inputs: dec_in LongTensor (B, L) = characters written so far (starting with <bos>);
        memory FloatTensor (B, N, d) = image features. Returns: logits FloatTensor (B, L, vocab).
        Logits at position t are the scores for the NEXT character after dec_in[:, t]."""
        # Q: What does it do and why does it exist?
        # A: It runs the decoder once: embeds the characters and positions, applies the causal mask and the
        #    decoder layers (attending to the image memory), and gives logits over the vocabulary.
        # Q: Why the causal mask and the padding mask?
        # A: The causal mask stops position t from seeing later characters, so the model cannot copy the answer
        #    in training. The padding mask makes attention ignore <pad> positions.
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
        # Q: What does it do and why does it exist?
        # A: It is the training / validation pass: encode the images, decode the shifted labels (teacher
        #    forcing) and compute the cross-entropy loss. It returns .loss and .logits.
        # Q: Why reshape to (B*L, vocab) and use ignore_index=-100?
        # A: cross_entropy expects one row of scores per position and one target id per row. Padded label
        #    positions hold -100 and are skipped, so they do not add loss.
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
        # Q: What does it do and why does it exist?
        # A: It predicts words for a batch of images with no true labels: encode the images once, then decode
        #    greedily or with beam search. The model's own predictions are used for validation and test.
        # Q: Why @torch.no_grad() and why is beam search done image by image?
        # A: No training happens here, so gradients would waste memory. _beam is written for a single image
        #    (memory of shape (1, N, d)), so the batch is looped; the results are then padded to one tensor.
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
        # Q: What does it do and why does it exist?
        # A: It writes the word one character at a time: start with <bos>, run the decoder, take the argmax of
        #    the last position, append it, and stop when every row has produced <eos>.
        # Q: Why masked_fill with PAD and a step limit of MAX_POS - 1?
        # A: A row that already ended must not get new characters, so it receives <pad>. The position table has
        #    only MAX_POS entries, so more steps than MAX_POS - 1 would index outside it.
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
        # Q: What does it do and why does it exist?
        # A: It does beam search for one image: it keeps the k best partial words, extends each with its k best
        #    next characters, keeps the k best results, and returns the best finished word.
        # Q: Why length-normalise the score and use log-probabilities?
        # A: Log-probabilities can be added to score a whole word, but longer words always have a lower sum.
        #    Dividing by the number of generated tokens removes that bias towards short words.
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
    # Q: What does it do and why does it exist?
    # A: It builds the model chosen by cfg.approach. For "finetune" it loads pre-trained TrOCR, replaces the
    #    decoder's embeddings and output layer with our character vocabulary and sets the special-token ids.
    #    For "custom_decoder" it returns CustomDecoderOCR on top of the pre-trained encoder.
    # Q: Why replace the embedding table and output layer, and why load_pretrained=False?
    # A: The pre-trained ones are made for English sub-word ids, so they are meaningless for our character ids.
    #    The other decoder layers keep their weights. With load_pretrained=False only the architecture is
    #    built, because evaluate.py loads our own trained weights next and needs no big download.
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
    # Q: What does it do and why does it exist?
    # A: It adds up the number of values (numel) of the model's parameters. It is used for the log line in
    #    train.py and for the "params" entry in the saved metrics.
    # Q: Why the trainable_only option?
    # A: With the default True, parameters with requires_grad=False (a frozen encoder) are left out, giving the
    #    number being trained. With False the full model size is reported, as done in checkpoints and results.
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def set_encoder_trainable(model, trainable):
    """Freeze (trainable=False) or unfreeze (True) the image encoder.
    Used with cfg.freeze_encoder_epochs: the first epochs can train only the decoder."""
    # Q: What does it do and why does it exist?
    # A: It sets requires_grad on every encoder parameter. train.run calls it at the start of every epoch to
    #    freeze the encoder in the first freeze_encoder_epochs epochs and to unfreeze it afterwards.
    # Q: Why would one freeze the encoder?
    # A: A frozen encoder is not updated, so the new decoder parts can settle first without gradients going
    #    into the pre-trained image features. The default config has freeze_encoder_epochs = 0 (no freezing).
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def new_param_names(model, cfg):
    """Names of parameters that are NEW (randomly initialised) and so get the larger lr_new.

    Returns a set of names from model.named_parameters(). train.make_optimizer uses it to split
    the parameters into two groups with two learning rates."""
    # Q: What does it do and why does it exist?
    # A: It returns the names of the parameters that start from random values (new decoder parts), so that
    #    train.make_optimizer can give them the larger learning rate lr_new.
    # Q: Why are the new parameters different for the two approaches?
    # A: In "custom_decoder" the whole decoder is new, so everything outside "encoder." counts. In "finetune"
    #    only the replaced character embeddings and the output layer are new; the rest is pre-trained.
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
    # Q: What does it do and why does it exist?
    # A: It calls model.generate with the right arguments for either approach and returns the predicted ids
    #    (starting with <bos>). train.evaluate_loader and evaluate.time_per_image use it.
    # Q: Why pass the special-token ids explicitly to the Hugging Face model?
    # A: Its built-in defaults are for the English tokenizer. Passing BOS, EOS and PAD makes generation use our
    #    ids. The custom model has its own generate() and takes only pixel_values, num_beams and max_length.
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
