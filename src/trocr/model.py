"""Model C: TrOCR-style Transformer OCR with a character-level Devanagari vocabulary.

Approach 1 ("finetune")       : pre-trained TrOCR-small encoder + pre-trained decoder whose token
                                embeddings / output layer are replaced by OUR character vocabulary.
Approach 2 ("custom_decoder") : pre-trained TrOCR-small encoder + a new nn.TransformerDecoder
                                (4 layers, d_model = encoder hidden size, 6 heads) trained from scratch.

Both expose the same two calls used by train.py / evaluate.py / predict.py:
    out = model(pixel_values=..., labels=...)          -> out.loss, out.logits
    ids = model.generate(pixel_values=..., num_beams=, max_length=)   -> LongTensor (B, L), starts with <bos>
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import BOS, EOS, IGNORE_INDEX, PAD

MAX_POS = 34  # learned positional embeddings of the custom decoder (>= max_length + 2)


# ----------------------------------------------------------------------------------------------
# Approach 2: custom decoder on top of the pre-trained encoder
# ----------------------------------------------------------------------------------------------
def shift_right(labels):
    """labels (…, -100 padded, ends with <eos>) -> decoder inputs: [<bos>, c1, …, cn], pad = <pad>."""
    ids = labels.new_full(labels.shape, PAD)
    ids[:, 0] = BOS
    ids[:, 1:] = labels[:, :-1]
    return ids.masked_fill(ids == IGNORE_INDEX, PAD)


class CustomDecoderOCR(nn.Module):
    is_custom = True

    def __init__(self, encoder, vocab_size, num_layers=4, heads=6, ff=1536, dropout=0.1):
        super().__init__()
        d = encoder.config.hidden_size
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_POS, d)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers, norm=nn.LayerNorm(d))
        self.head = nn.Linear(d, vocab_size)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    def encode_images(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state

    def decode(self, dec_in, memory):
        L = dec_in.shape[1]
        x = self.embed(dec_in) + self.pos(torch.arange(L, device=dec_in.device))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dec_in.device), 1)
        h = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=(dec_in == PAD) & (dec_in != BOS))
        return self.head(h)

    def forward(self, pixel_values, labels=None, **_):
        memory = self.encode_images(pixel_values)
        logits = self.decode(shift_right(labels), memory)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX)
        return types.SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, pixel_values, num_beams=1, max_length=32, **_):
        memory = self.encode_images(pixel_values)
        if num_beams <= 1:
            return self._greedy(memory, max_length)
        outs = [self._beam(memory[i: i + 1], num_beams, max_length) for i in range(memory.shape[0])]
        width = max(len(o) for o in outs)
        res = torch.full((len(outs), width), PAD, dtype=torch.long, device=memory.device)
        for i, o in enumerate(outs):
            res[i, : len(o)] = torch.tensor(o, device=memory.device)
        return res

    def _greedy(self, memory, max_length):
        B = memory.shape[0]
        ids = torch.full((B, 1), BOS, dtype=torch.long, device=memory.device)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        for _ in range(min(max_length, MAX_POS - 1)):
            nxt = self.decode(ids, memory)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ids

    def _beam(self, memory, k, max_length):
        """Beam search for one image; returns the best id list (starting with <bos>)."""
        dev = memory.device
        beams = [([BOS], 0.0)]
        finished = []
        for _ in range(min(max_length, MAX_POS - 1)):
            seqs = torch.tensor([b[0] for b in beams], device=dev)
            logp = F.log_softmax(self.decode(seqs, memory.expand(len(beams), -1, -1))[:, -1], -1)
            cand = []
            for bi, (seq, score) in enumerate(beams):
                top = logp[bi].topk(k)
                cand += [(seq + [t.item()], score + v.item()) for v, t in zip(top.values, top.indices)]
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in cand:
                if seq[-1] == EOS:
                    finished.append((seq, score / (len(seq) - 1)))  # length-normalised
                else:
                    beams.append((seq, score))
                if len(beams) == k:
                    break
            if not beams or (len(finished) >= k and max(f[1] for f in finished) > beams[0][1] / len(beams[0][0])):
                break
        if not finished:
            finished = [(s, sc / max(len(s) - 1, 1)) for s, sc in beams]
        return max(finished, key=lambda f: f[1])[0]


# ----------------------------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------------------------
def build_model(cfg, tokenizer, load_pretrained=True):
    """Create the model for cfg.approach. load_pretrained=False builds the same architecture with
    random weights (used when restoring one of our own checkpoints)."""
    from transformers import AutoModel, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    n = len(tokenizer)
    if load_pretrained:
        base = VisionEncoderDecoderModel.from_pretrained(cfg.pretrained)
    else:
        base = VisionEncoderDecoderModel(VisionEncoderDecoderConfig.from_pretrained(cfg.pretrained))

    if cfg.approach == "custom_decoder":
        return CustomDecoderOCR(base.encoder, n)
    if cfg.approach != "finetune":
        raise ValueError(f"unknown approach {cfg.approach!r} (use 'finetune' or 'custom_decoder')")

    # ---- Approach 1: replace the English sub-word vocabulary by our character vocabulary ----
    dec = base.decoder
    dec.resize_token_embeddings(n)
    emb = dec.get_input_embeddings()
    emb.padding_idx = PAD  # the English model used padding_idx=1, which is <bos> for us
    nn.init.normal_(emb.weight, std=0.02)
    with torch.no_grad():
        emb.weight[PAD].zero_()
    out = dec.get_output_embeddings()
    if out.out_features != n:  # not tied / not resized: replace the output layer
        head = nn.Linear(out.in_features, n, bias=False)
        dec.set_output_embeddings(head)
        out = head
    if out.weight.data_ptr() != emb.weight.data_ptr():
        nn.init.normal_(out.weight, std=0.02)

    for c in (base.config, base.config.decoder):
        c.decoder_start_token_id, c.pad_token_id, c.eos_token_id, c.bos_token_id = BOS, PAD, EOS, BOS
    base.config.vocab_size = n
    base.config.decoder.vocab_size = n
    g = base.generation_config
    g.decoder_start_token_id, g.pad_token_id, g.eos_token_id, g.bos_token_id = BOS, PAD, EOS, BOS
    g.max_length, g.num_beams, g.early_stopping, g.use_cache = cfg.max_length, cfg.num_beams, True, True
    return base


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def new_param_names(model, cfg):
    """Names of parameters that are NEW (randomly initialised) and so get the larger lr_new."""
    if cfg.approach == "custom_decoder":
        return {n for n, _ in model.named_parameters() if not n.startswith("encoder.")}
    return {n for n, _ in model.named_parameters()
            if "decoder.model.decoder.embed_tokens" in n or "decoder.output_projection" in n}


@torch.no_grad()
def generate_ids(model, pixel_values, num_beams=1, max_length=32):
    """Unified generation for both approaches. Returns ids starting with <bos>."""
    kw = dict(pixel_values=pixel_values, num_beams=num_beams, max_length=max_length)
    if not getattr(model, "is_custom", False):
        kw.update(decoder_start_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=BOS, use_cache=True)
        if num_beams > 1:
            kw["early_stopping"] = True
    return model.generate(**kw)


if __name__ == "__main__":
    from .config import load_config
    from .tokenizer import CharTokenizer

    tok = CharTokenizer(list("कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसहािीुूेैोौंँ्"))
    for approach in ("finetune", "custom_decoder"):
        cfg = load_config(None, {"approach": approach})
        model = build_model(cfg, tok)
        x = torch.randn(2, 3, 384, 384)
        labels = tok.batch_encode(["प्रेम", "धर्म"])
        out = model(pixel_values=x, labels=labels)
        print(f"{approach}: params={count_params(model) / 1e6:.1f}M  loss={out.loss.item():.3f} "
              f"(~ ln({len(tok)}) = {torch.log(torch.tensor(float(len(tok)))):.2f} at init)")
