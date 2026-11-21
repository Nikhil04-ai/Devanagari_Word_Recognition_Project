"""Config handling for the TrOCR package: YAML file -> attribute-style object, with defaults.

Where it fits in the pipeline: every other TrOCR file (dataset, model, train, evaluate) reads its
settings from the object that `load_config` returns. So all hyper-parameters live in one place.

How a config is built (later steps win over earlier ones):
    1. DEFAULTS below                     (safe values that always exist)
    2. the YAML file, if a path is given  (for example configs/trocr/full.yaml)
    3. command-line overrides             (for example --epochs 1 or --set lr_new=1e-3)
The result is a SimpleNamespace, so code can write `cfg.epochs` instead of `cfg["epochs"]`.
"""
# SimpleNamespace turns a dict into an object with attributes (cfg.epochs).
from types import SimpleNamespace

# PyYAML reads the .yaml config files and also converts text like "1e-3" or "20" to numbers.
import yaml

# Default value of every allowed setting. A YAML file or an override may only use keys listed here
# (unknown keys are rejected in load_config), so a typo in a config file cannot pass silently.
DEFAULTS = dict(
    approach="finetune",              # "finetune" (pre-trained decoder, new vocab) | "custom_decoder" (backup)
    # Hugging Face name of the pre-trained TrOCR model (its image encoder is always reused).
    pretrained="microsoft/trocr-small-handwritten",
    # Folder that holds train.csv, val.csv, test.csv and vocab.txt (made by the data preparation step).
    data_dir="data/processed/words",
    train_csv="train.csv",            # file name of the training manifest inside data_dir
    val_csv="val.csv",                # file name of the validation manifest inside data_dir
    image_root=".",                   # image_path in the CSVs is relative to the repository root
    train_limit=0,                    # >0: train on a random subset of N images (lever if the run is too slow)
    overfit_n=0,                      # >0: train and validate on the first N training samples only
    val_limit=0,                      # >0: validate on a random subset of N validation images (faster epochs)
    batch_size=16,                    # images per forward pass (per mini-batch)
    # Gradient accumulation: gradients of this many mini-batches are added up before one optimizer
    # step. Effective batch size = batch_size * grad_accum = 32 here, with the memory cost of only 16.
    grad_accum=2,
    epochs=10,                        # maximum number of passes over the training set
    # Two learning rates: the pre-trained weights only need a small update (lr_pretrained), while the
    # new random weights (our character embeddings and output layer) must learn from scratch, so
    # they get a larger step (lr_new).
    lr_pretrained=5e-5,
    lr_new=3e-4,
    weight_decay=0.01,                # AdamW weight decay (a small penalty on large weights)
    warmup_steps=500,                 # optimizer steps of linear learning-rate warm-up (see make_scheduler)
    grad_clip=1.0,                    # maximum gradient norm; larger gradients are scaled down
    max_length=32,                    # maximum label length in tokens, including <eos>
    num_beams=4,                      # only used by `evaluate --decoding beam`
    freeze_encoder_epochs=0,          # number of first epochs in which the image encoder is frozen
    patience=5,                       # early stopping on val CER
    max_minutes=0,                    # time budget: stops cleanly and keeps best.pt (0 = no limit)
    num_workers=2,                    # DataLoader worker processes that load images in parallel
    seed=42,                          # random seed, for reproducible runs
    ckpt_dir="checkpoints/trocr_full",  # folder where last.pt, best.pt and log.csv are written
    run_name="trocr_full",            # name of this run
)


def load_config(path=None, overrides=None):
    """Load a YAML config (or just defaults), apply `overrides` (dict, None values ignored), return a namespace.

    Inputs: `path` = YAML file path or None; `overrides` = dict of setting -> value or None.
    Returns: SimpleNamespace with every key of DEFAULTS as an attribute.
    Why: one function gives all other files the same, checked settings object."""
    # Start from a copy so that the DEFAULTS dict itself is never changed.
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            # `or {}` handles an empty YAML file (safe_load returns None for it).
            cfg.update(yaml.safe_load(f) or {})
    # None means "the user did not set this flag", so it must not overwrite the YAML or default value.
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    # Reject keys that are not in DEFAULTS. This catches typos such as "epoch" instead of "epochs".
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return SimpleNamespace(**cfg)


def parse_overrides(items):
    """['epochs=1', 'overfit_n=20'] -> {'epochs': 1, 'overfit_n': 20} using YAML typing.

    Inputs: list of "key=value" strings (from the `--set` command-line option), or None.
    Returns: dict of key -> value, where each value has the right Python type (int, float, bool, str).
    Why YAML typing: yaml.safe_load("1") gives the int 1 and yaml.safe_load("1e-3") gives a float,
    so we do not need our own number parsing."""
    out = {}
    for item in items or []:
        # partition splits at the FIRST "=", so a value may itself contain "=".
        k, _, v = item.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def to_dict(cfg):
    """Config namespace -> plain dict. Used to store the config inside a checkpoint file
    (train.py) so that evaluate.py can rebuild the same model later."""
    return dict(vars(cfg))
