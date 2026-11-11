"""Config handling for the TrOCR package: YAML file -> attribute-style object, with defaults."""
from types import SimpleNamespace

import yaml

DEFAULTS = dict(
    approach="finetune",              # "finetune" (pre-trained decoder, new vocab) | "custom_decoder" (backup)
    pretrained="microsoft/trocr-small-handwritten",
    data_dir="data/processed/words",
    train_csv="train.csv",
    val_csv="val.csv",
    image_root=".",                   # image_path in the CSVs is relative to the repository root
    train_limit=0,                    # >0: train on a random subset of N images (lever if the run is too slow)
    overfit_n=0,                      # >0: train and validate on the first N training samples only
    val_limit=0,                      # >0: validate on a random subset of N validation images (faster epochs)
    batch_size=16,
    grad_accum=2,
    epochs=10,
    lr_pretrained=5e-5,
    lr_new=3e-4,
    weight_decay=0.01,
    warmup_steps=500,
    grad_clip=1.0,
    max_length=32,
    num_beams=4,                      # only used by `evaluate --decoding beam`
    freeze_encoder_epochs=0,
    patience=5,                       # early stopping on val CER
    max_minutes=0,                    # time budget: stops cleanly and keeps best.pt (0 = no limit)
    num_workers=2,
    seed=42,
    ckpt_dir="checkpoints/trocr_full",
    run_name="trocr_full",
)


def load_config(path=None, overrides=None):
    """Load a YAML config (or just defaults), apply `overrides` (dict, None values ignored), return a namespace."""
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return SimpleNamespace(**cfg)


def parse_overrides(items):
    """['epochs=1', 'overfit_n=20'] -> {'epochs': 1, 'overfit_n': 20} using YAML typing."""
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def to_dict(cfg):
    return dict(vars(cfg))
