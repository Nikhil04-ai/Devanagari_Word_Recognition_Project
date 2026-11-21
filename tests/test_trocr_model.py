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


def tiny_model():
    """Build a very small CustomDecoderOCR (random weights). Returns (model, tokenizer).
    Encoder: ViT with hidden size 48, 1 layer, 3 heads, 32x32 images in 16x16 patches (4 patches).
    Decoder: 1 layer, 3 heads (48 / 3 = 16 per head), feed-forward width 64."""
    from transformers import ViTConfig, ViTModel

    enc = ViTModel(ViTConfig(hidden_size=48, num_hidden_layers=1, num_attention_heads=3,
                             intermediate_size=64, image_size=32, patch_size=16), add_pooling_layer=False)
    tok = CharTokenizer(list("कखगाि"))
    m = CustomDecoderOCR(enc, len(tok), num_layers=1, heads=3, ff=64)
    return m, tok


def test_shift_right():
    """The decoder input is <bos> followed by the labels moved right; the last label slot is dropped."""
    labels = torch.tensor([[7, 8, EOS], [7, EOS, -100]])
    # Row 0: [7, 8, EOS] -> [BOS, 7, 8]. Row 1: [7, EOS, -100] -> [BOS, 7, EOS] (the -100 is shifted out).
    assert shift_right(labels).tolist() == [[BOS, 7, 8], [BOS, 7, EOS]]
    assert PAD == 0


def test_custom_forward_and_generate():
    """forward gives a finite scalar loss; generate returns (B, L) ids that start with <bos>, for greedy and beam search."""
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
    cfg = load_config(None, parse_overrides(["epochs=3", "overfit_n=20"]))
    assert cfg.epochs == 3 and cfg.overfit_n == 20


def test_unknown_config_key_rejected():
    """A key that is not in DEFAULTS must raise KeyError (protects against typos)."""
    import pytest

    with pytest.raises(KeyError):
        load_config(None, {"augment": True})


def test_shipped_configs_load():
    """The YAML files in configs/trocr/ must load without errors, use the finetune approach and allow labels of 24+ tokens.
    Needs to be run from the repository root because the path is relative."""
    for name in ("full", "overfit"):
        cfg = load_config(f"configs/trocr/{name}.yaml")
        assert cfg.approach == "finetune" and cfg.max_length >= 24
