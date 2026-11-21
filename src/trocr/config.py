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

# ======================================= CONCEPTS USED IN THIS FILE =======================================
# YAML: a simple text format for settings ("epochs: 10"). Read in load_config from configs/trocr/*.yaml.
# Hyper-parameter: a setting that we choose before training (learning rate, batch size...), not learned
#     by the model. All of them are listed in DEFAULTS.
# Config layering: defaults < YAML file < command-line overrides; the last one that sets a key wins.
#     Done in load_config with cfg.update(...) twice.
# SimpleNamespace: a tiny object whose attributes come from a dict, so we write cfg.epochs. Built at the end
#     of load_config.
# Command-line override (--set KEY=VALUE): changes one setting for one run without editing the YAML.
#     Turned into a dict by parse_overrides.
# Batch size / gradient accumulation: batch_size images per forward pass; gradients of grad_accum batches are
#     added before one optimizer step, so effective batch = batch_size * grad_accum (see DEFAULTS).
# Two learning rates: lr_pretrained (small, for pre-trained weights) and lr_new (large, for new random
#     weights). They are used in train.make_optimizer; here they are only stored.
# Warm-up / weight decay / gradient clipping: warmup_steps = lr rises slowly at the start; weight_decay = small
#     penalty on big weights (AdamW); grad_clip = maximum gradient norm. All three are only stored here.
# Early stopping (patience): stop if validation CER does not improve for `patience` epochs.
# Beam search (num_beams): decoding that keeps several best partial words; used only by evaluate --decoding beam.
# Seed: a fixed number that makes random choices repeatable (seed=42 in DEFAULTS).
# Overfit test (overfit_n): train on a tiny set to prove the model can memorise it (a sanity check).
# Checkpoint (ckpt_dir): a saved file of weights; to_dict stores the config inside it.
# ==========================================================================================================

# ======================================= FILE-LEVEL Q&A (VIVA) ============================================
# Q: What part of the project is this file?
# A: The settings loader of the TrOCR package. dataset.py, model.py, train.py and evaluate.py all get their
#    hyper-parameters from the object returned by load_config, so every setting lives in one place.
# Q: How is it used, and what does it read and write?
# A: It is imported, not run. It reads an optional YAML file (for example configs/trocr/full.yaml) and a dict of
#    overrides, and returns a SimpleNamespace. It writes nothing to disk.
# Q: Why reject unknown keys?
# A: A typo such as "epoch" instead of "epochs" would otherwise be ignored silently and the run would use the
#    default value. load_config raises KeyError instead.
# ==========================================================================================================

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
    # Q: What does it do and why does it exist?
    # A: It merges defaults, the YAML file and the overrides (later ones win), rejects unknown keys and
    #    returns a namespace. All TrOCR files call it to get one checked settings object.
    # Q: Why are None overrides ignored and unknown keys rejected?
    # A: argparse gives None for flags the user did not pass; if None were applied it would erase the YAML or
    #    default value. Unknown keys are rejected so a typo cannot pass silently.
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
    # Q: What does it do and why does it exist?
    # A: It turns the --set strings ("epochs=1") into a dict with typed values, so load_config can use them
    #    as overrides.
    # Q: Why yaml.safe_load for the value, and why partition instead of split?
    # A: safe_load already gives int, float, bool or str, so no own number parsing is needed. partition cuts
    #    only at the first "=", so a value that contains "=" does not break.
    out = {}
    for item in items or []:
        # partition splits at the FIRST "=", so a value may itself contain "=".
        k, _, v = item.partition("=")
        out[k.strip()] = yaml.safe_load(v)
    return out


def to_dict(cfg):
    """Config namespace -> plain dict. Used to store the config inside a checkpoint file
    (train.py) so that evaluate.py can rebuild the same model later."""
    # Q: What does it do and why does it exist?
    # A: It converts the config namespace back to a plain dict. train.py stores that dict inside every
    #    checkpoint file.
    # Q: Why store the config in the checkpoint?
    # A: evaluate.py then rebuilds the model with the same settings it was trained with, even if the YAML file
    #    was edited later. vars(cfg) is the namespace's attribute dict; dict() makes a copy.
    return dict(vars(cfg))
