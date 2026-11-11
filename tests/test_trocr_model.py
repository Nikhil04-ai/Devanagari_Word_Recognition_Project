"""Offline tests for the TrOCR pieces (tiny random encoder, no downloads)."""
import torch

from src.trocr.config import load_config, parse_overrides
from src.trocr.model import CustomDecoderOCR, shift_right
from src.trocr.tokenizer import BOS, EOS, PAD, CharTokenizer
from src.trocr.train import make_optimizer, make_scheduler


def tiny_model():
    from transformers import ViTConfig, ViTModel

    enc = ViTModel(ViTConfig(hidden_size=48, num_hidden_layers=1, num_attention_heads=3,
                             intermediate_size=64, image_size=32, patch_size=16), add_pooling_layer=False)
    tok = CharTokenizer(list("कखगाि"))
    m = CustomDecoderOCR(enc, len(tok), num_layers=1, heads=3, ff=64)
    return m, tok


def test_shift_right():
    labels = torch.tensor([[7, 8, EOS], [7, EOS, -100]])
    assert shift_right(labels).tolist() == [[BOS, 7, 8], [BOS, 7, EOS]]
    assert PAD == 0


def test_custom_forward_and_generate():
    m, tok = tiny_model()
    x = torch.randn(2, 3, 32, 32)
    labels = tok.batch_encode(["कख", "गाि"])
    out = m(pixel_values=x, labels=labels)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    m.eval()
    for beams in (1, 3):
        ids = m.generate(pixel_values=x, num_beams=beams, max_length=6)
        assert ids.shape[0] == 2 and (ids[:, 0] == BOS).all()
        assert isinstance(tok.decode(ids[0]), str)


def test_custom_overfits_one_batch():
    torch.manual_seed(0)
    m, tok = tiny_model()
    x = torch.randn(2, 3, 32, 32)
    labels = tok.batch_encode(["कख", "गाि"])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = m(pixel_values=x, labels=labels).loss.item()
    for _ in range(150):
        opt.zero_grad()
        m(pixel_values=x, labels=labels).loss.backward()
        opt.step()
    assert m(pixel_values=x, labels=labels).loss.item() < 0.2 * first


def test_optimizer_groups_and_scheduler():
    m, _ = tiny_model()
    cfg = load_config(None, {"approach": "custom_decoder"})
    opt = make_optimizer(m, cfg)
    assert [g["lr"] for g in opt.param_groups] == [cfg.lr_pretrained, cfg.lr_new]
    sched = make_scheduler(opt, warmup=10, total=100)
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[1]["lr"])
    assert lrs[0] < lrs[9] and lrs[-1] < 0.05 * cfg.lr_new  # warm-up up, cosine down


def test_config_overrides():
    cfg = load_config(None, parse_overrides(["epochs=3", "overfit_n=20"]))
    assert cfg.epochs == 3 and cfg.overfit_n == 20


def test_unknown_config_key_rejected():
    import pytest

    with pytest.raises(KeyError):
        load_config(None, {"augment": True})


def test_shipped_configs_load():
    for name in ("full", "overfit"):
        cfg = load_config(f"configs/trocr/{name}.yaml")
        assert cfg.approach == "finetune" and cfg.max_length >= 24
