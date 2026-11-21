"""Offline tests for the TrOCR pieces (tiny random encoder, no downloads).

What is tested: shift_right (teacher-forcing input), the custom decoder's forward / generate
(greedy and beam), a tiny overfit check, the two-learning-rate optimizer with the warm-up +
cosine scheduler, and the config helpers (overrides, unknown keys, shipped YAML files).
Why a tiny random ViT encoder: the real pre-trained model needs a download and is slow.
Run with: pytest tests/test_trocr_model.py
"""
import torch

# Code under test: config, model pieces, tokenizer and the optimizer / scheduler builders.
from src.trocr.config import load_config, parse_overrides
from src.trocr.model import CustomDecoderOCR, shift_right
from src.trocr.tokenizer import BOS, EOS, PAD, CharTokenizer
from src.trocr.train import make_optimizer, make_scheduler

# ======================================================================================================
# CONCEPTS USED IN THIS FILE
# ======================================================================================================
# unit test / assert: a small function that checks one behaviour; assert is a line that must be true or the test fails.
# synthetic (fake) data: random images from torch.randn and a hand-made labels tensor. No real images or dataset.
# random weights / tiny model: tiny_model() builds a very small network that is not trained. It is fast and needs
#     no download, and it is enough to test the shapes and logic of the code.
# transformers package: the Hugging Face library. tiny_model() imports ViTConfig and ViTModel from it to make the
#     image encoder, so the tests that call tiny_model() need it installed.
# ViT (Vision Transformer) encoder: cuts the image into patches (here 16x16) and gives one feature vector per patch
#     (plus a special token). Here: hidden size 48, 1 layer, 3 attention heads, 32x32 images -> 4 patches.
# encoder-decoder: the encoder reads the image into features; the decoder writes the text one character at a time
#     while looking at those features (cross-attention). CustomDecoderOCR is this model with our own small decoder.
# special tokens pad/bos/eos: PAD=0 filler, BOS=1 start of the decoder input, EOS=2 end of the word.
# label masking -100: padded label slots hold -100; cross-entropy ignores them, so padding adds no loss.
# teacher forcing: while training, the decoder gets the TRUE previous characters as input, not its own guesses.
#     shift_right builds that input: [<bos>, c1, c2, ...] from the labels [c1, c2, ..., <eos>].
# logits / cross-entropy loss: logits are the scores for each vocabulary token; cross-entropy turns them and the
#     labels into one number (the loss) that is lower when the right characters get high scores.
# scalar loss: a loss with ndim == 0, i.e. a single number. torch.isfinite checks it is not NaN or infinity.
# greedy generation: num_beams=1; at each step pick the single most likely next character.
# beam search: num_beams>1; keep the k best partial words at each step and choose the best finished one.
# generate() output: ids of shape (B, L) that always start with <bos>; short rows are padded with <pad>.
# overfitting one batch: training on one small batch until the loss is almost zero. A healthy model can memorise
#     it; if it cannot, the forward pass, loss or masking is broken.
# AdamW / learning rate: the optimizer that updates the weights, and the step size it uses.
# parameter groups / two learning rates: pre-trained encoder weights get lr_pretrained (small); new random weights
#     (decoder, embeddings, head) get lr_new (larger). They are two groups in one optimizer.
# scheduler / warm-up / cosine decay: the learning rate rises linearly for the first warmup steps, then falls
#     along a cosine curve towards 0. make_scheduler builds this with torch LambdaLR.
# config / YAML / overrides: settings live in a YAML file; load_config reads it over the DEFAULTS and applies
#     overrides given as key=value text, which parse_overrides converts to numbers or strings.
# KeyError / pytest.raises: load_config raises KeyError for an unknown key; pytest.raises passes only if it happens.
# torch.manual_seed: fixes the random numbers, so the overfit test gives the same result each run.

# ======================================================================================================
# Q&A ABOUT THIS FILE
# ======================================================================================================
# Q: What part of the project does this file protect? A: The TrOCR model side: src/trocr/model.py (shift_right,
#    CustomDecoderOCR forward / generate), the optimizer and scheduler in src/trocr/train.py, and the config
#    loader in src/trocr/config.py (plus the shipped YAML files in configs/trocr/).
# Q: How is it run? A: From the repository root with: pytest tests/test_trocr_model.py (the root matters because
#    test_shipped_configs_load uses relative paths). It needs the transformers package and torch installed.
# Q: Why is no real dataset needed? A: The tests use random image tensors and tiny hand-made labels, because they
#    check shapes, loss, masking and learning-rate logic, not recognition accuracy.
# Q: Why does it need the transformers package? A: tiny_model() builds the image encoder with ViTModel and
#    ViTConfig from transformers. Without it, tests that call tiny_model() fail with ImportError (the import is
#    inside the function, so the file can still be imported and the tests that do not need it still run).


def tiny_model():
    """Build a very small CustomDecoderOCR (random weights). Returns (model, tokenizer).
    Encoder: ViT with hidden size 48, 1 layer, 3 heads, 32x32 images in 16x16 patches (4 patches).
    Decoder: 1 layer, 3 heads (48 / 3 = 16 per head), feed-forward width 64."""
    # Q: What does this helper do? A: It builds a very small CustomDecoderOCR with random weights and returns
    #    (model, tokenizer). The tokenizer has 5 characters, so the vocabulary has 9 tokens (4 special + 5).
    # Q: Why is it small and random? A: The real pre-trained encoder needs a download and is slow. These tests check
    #    logic, not quality, so a tiny encoder (hidden 48, 1 layer, 3 heads, 32x32 images in 16x16 patches) is enough.
    # Q: Why must the sizes match? A: The decoder width is taken from encoder.config.hidden_size (48), and 48 must be
    #    divisible by the number of heads (3). Both encoder and decoder use 3 heads, giving 16 per head.
    # Q: Why is the transformers import inside the function? A: So importing this test file works without
    #    transformers; only tests that need the model fail if it is missing.
    from transformers import ViTConfig, ViTModel

    enc = ViTModel(ViTConfig(hidden_size=48, num_hidden_layers=1, num_attention_heads=3,
                             intermediate_size=64, image_size=32, patch_size=16), add_pooling_layer=False)
    tok = CharTokenizer(list("कखगाि"))
    m = CustomDecoderOCR(enc, len(tok), num_layers=1, heads=3, ff=64)
    return m, tok


def test_shift_right():
    """The decoder input is <bos> followed by the labels moved right; the last label slot is dropped."""
    # Q: What does this test check? A: shift_right turns labels into decoder input: <bos> first, then the labels moved
    #    one place right, and the last label slot is dropped. [7, 8, EOS] -> [BOS, 7, 8];
    #    [7, EOS, -100] -> [BOS, 7, EOS].
    # Q: Why does it matter? What bug would make it fail? A: This is teacher forcing. If the shift were off by one,
    #    the decoder would see the answer it must predict (cheating) or the wrong character, and the model would
    #    not learn. Also, -100 must never reach the embedding layer. This test also checks that PAD is 0.
    # Q: How is the input built and why is it enough? A: A hand-made 2x3 tensor of ids. Row 0 is a full row; row 1 has
    #    a -100 pad at the end, which falls in the dropped last slot. 7 and 8 stand for any two character ids.
    # Q: Which project function does it exercise? A: shift_right in src/trocr/model.py. It needs no model, so it
    #    runs fast.
    labels = torch.tensor([[7, 8, EOS], [7, EOS, -100]])
    # Row 0: [7, 8, EOS] -> [BOS, 7, 8]. Row 1: [7, EOS, -100] -> [BOS, 7, EOS] (the -100 is shifted out).
    assert shift_right(labels).tolist() == [[BOS, 7, 8], [BOS, 7, EOS]]
    assert PAD == 0


def test_custom_forward_and_generate():
    """forward gives a finite scalar loss; generate returns (B, L) ids that start with <bos>, for greedy and beam search."""
    # Q: What does this test check? A: forward on 2 random images returns a loss that is a single finite number; and
    #    generate, with greedy (num_beams=1) and beam search (num_beams=3), returns 2 rows that all start with <bos>
    #    and can be decoded to a str.
    # Q: Why does it matter? What bug would make it fail? A: It proves the whole pipeline runs end to end: encoder ->
    #    decoder -> loss, and the two decoding methods. It would fail on a shape mismatch, a NaN loss, a broken
    #    causal mask / label shift, or if generate did not start with <bos> (evaluation code relies on this).
    # Q: How is the input built and why is it enough? A: torch.randn(2, 3, 32, 32) is a random batch of 2 images
    #    matching the tiny encoder; labels come from tok.batch_encode(["कख", "गाि"]) (different lengths, so -100 padding is
    #    used too). The test checks shapes and validity only, not the text, because the weights are random.
    # Q: Which project functions does it exercise? A: CustomDecoderOCR.forward, generate, _greedy, _beam, decode
    #    and encode_images (src/trocr/model.py), plus CharTokenizer.batch_encode and decode.
    m, tok = tiny_model()
    # Random images: batch 2, 3 channels, 32x32.
    x = torch.randn(2, 3, 32, 32)
    labels = tok.batch_encode(["कख", "गाि"])
    out = m(pixel_values=x, labels=labels)
    # ndim == 0 means the loss is a single number.
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    m.eval()
    # num_beams=1 -> greedy; num_beams=3 -> beam search.
    for beams in (1, 3):
        ids = m.generate(pixel_values=x, num_beams=beams, max_length=6)
        assert ids.shape[0] == 2 and (ids[:, 0] == BOS).all()
        assert isinstance(tok.decode(ids[0]), str)


def test_custom_overfits_one_batch():
    """The model must be able to memorise one tiny batch: after 150 steps the loss is below 20% of the first loss.
    If this fails, something in forward / loss / masking is broken."""
    # Q: What does this test check? A: Training on the same 2 images and labels for 150 AdamW steps (lr 3e-3) brings
    #    the loss below 20% of the first loss.
    # Q: Why does it matter? What bug would make it fail? A: It is the standard sanity check: a correct model can
    #    memorise a tiny batch. If the loss does not fall, something is broken: the label shift, the loss masking
    #    with -100, the causal mask, gradients not reaching the weights, or the decoder not seeing the image.
    # Q: How is the input built and why is it enough? A: One fixed random batch (torch.manual_seed(0) makes it
    #    repeatable) and two short labels. The test does not need real data, because memorising does not depend
    #    on what the images look like; random images are different from each other, so the model can tell them apart.
    # Q: Which project functions does it exercise? A: CustomDecoderOCR.forward (loss, backward) with the tiny
    #    model; the optimizer here is plain torch AdamW, not make_optimizer.
    torch.manual_seed(0)
    m, tok = tiny_model()
    x = torch.randn(2, 3, 32, 32)
    labels = tok.batch_encode(["कख", "गाि"])
    # A fairly high lr is fine here because the model is tiny and the goal is only to overfit.
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = m(pixel_values=x, labels=labels).loss.item()
    for _ in range(150):
        opt.zero_grad()
        m(pixel_values=x, labels=labels).loss.backward()
        opt.step()
    assert m(pixel_values=x, labels=labels).loss.item() < 0.2 * first


def test_optimizer_groups_and_scheduler():
    """make_optimizer gives 2 groups (pre-trained lr, new lr); the scheduler warms up and then decays."""
    # Q: What does this test check? A: make_optimizer builds two parameter groups with lrs [lr_pretrained, lr_new],
    #    and make_scheduler warms up (lr at step 0 < lr at step 9) and then decays (final lr < 5% of lr_new).
    # Q: Why does it matter? What bug would make it fail? A: The pre-trained encoder needs a small step and the new
    #    random layers a bigger one. If the groups were mixed up, training would damage the encoder or learn too
    #    slowly. A wrong scheduler (no warm-up, no decay) would make training unstable or stop too early.
    # Q: How is the input built and why is it enough? A: The tiny model with approach "custom_decoder" (the encoder is
    #    old, the rest is new, so both groups exist). The scheduler is run for 100 steps with warmup=10 and total=100.
    #    opt.step() runs with no gradients, so it changes no weights; only the lr schedule is being observed.
    # Q: Which project functions does it exercise? A: make_optimizer and make_scheduler (src/trocr/train.py),
    #    new_param_names (src/trocr/model.py) and load_config (src/trocr/config.py).
    m, _ = tiny_model()
    cfg = load_config(None, {"approach": "custom_decoder"})
    opt = make_optimizer(m, cfg)
    # Group 0 = encoder (lr_pretrained), group 1 = new decoder / head (lr_new).
    assert [g["lr"] for g in opt.param_groups] == [cfg.lr_pretrained, cfg.lr_new]
    sched = make_scheduler(opt, warmup=10, total=100)
    lrs = []
    for _ in range(100):
        # opt.step() before sched.step() is the normal order (no gradients exist, so it changes nothing).
        opt.step()
        sched.step()
        # Record the lr of the "new" group after each step.
        lrs.append(opt.param_groups[1]["lr"])
    # lr at step 0 is smaller than at step 9 (warm-up), and at the end it is under 5% of lr_new (cosine decay).
    assert lrs[0] < lrs[9] and lrs[-1] < 0.05 * cfg.lr_new  # warm-up up, cosine down


def test_config_overrides():
    """parse_overrides converts "key=value" text to typed values, and load_config applies them."""
    # Q: What does this test check? A: parse_overrides(["epochs=3", "overfit_n=20"]) gives numbers (not text), and
    #    load_config applies them, so cfg.epochs == 3 and cfg.overfit_n == 20.
    # Q: Why does it matter? What bug would make it fail? A: Command-line overrides (--set key=value) must change the
    #    run. If values stayed strings ("3"), later code such as range(cfg.epochs) would crash; if overrides were
    #    ignored, a quick test run would silently use the default 10 epochs.
    # Q: How is the input built and why is it enough? A: Two short key=value strings, one per kind of key. No YAML
    #    file is needed (path=None), so load_config uses only DEFAULTS plus the overrides.
    # Q: Which project functions does it exercise? A: parse_overrides and load_config in src/trocr/config.py.
    cfg = load_config(None, parse_overrides(["epochs=3", "overfit_n=20"]))
    assert cfg.epochs == 3 and cfg.overfit_n == 20


def test_unknown_config_key_rejected():
    """A key that is not in DEFAULTS must raise KeyError (protects against typos)."""
    # Q: What does this test check? A: load_config raises KeyError when an override uses a key that is not in DEFAULTS
    #    ("augment" is not a TrOCR setting).
    # Q: Why does it matter? What bug would make it fail? A: A typo such as "epoch" instead of "epochs" would otherwise
    #    be accepted and silently ignored, so the run would not do what the user thinks. It fails if the check in
    #    load_config is removed.
    # Q: How is the input built and why is it enough? A: One dict with a single key that does not exist. One bad key
    #    is enough to trigger the "unknown keys" rule; pytest.raises makes the test pass only if KeyError appears.
    # Q: Which project function does it exercise? A: load_config (the unknown-key check) in src/trocr/config.py.
    import pytest

    with pytest.raises(KeyError):
        load_config(None, {"augment": True})


def test_shipped_configs_load():
    """The YAML files in configs/trocr/ must load without errors, use the finetune approach and allow labels of 24+ tokens.
    Needs to be run from the repository root because the path is relative."""
    # Q: What does this test check? A: The real files configs/trocr/full.yaml and overfit.yaml load without error,
    #    use approach "finetune", and have max_length of at least 24.
    # Q: Why does it matter? What bug would make it fail? A: A typo or unknown key in a YAML file would only show up
    #    when a long (GPU) run starts. This catches it in seconds. max_length >= 24 matters because labels can be up to
    #    23 characters long, and one more slot is needed for <eos>; a smaller value would cut real words.
    # Q: How is the input built and why is it enough? A: No fake input: it reads the real shipped YAML files, because
    #    those are exactly what must be valid. It must run from the repository root since the paths are relative.
    # Q: Which project function does it exercise? A: load_config with a YAML path (src/trocr/config.py).
    for name in ("full", "overfit"):
        cfg = load_config(f"configs/trocr/{name}.yaml")
        assert cfg.approach == "finetune" and cfg.max_length >= 24
