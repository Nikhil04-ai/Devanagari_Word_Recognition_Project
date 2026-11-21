"""Train the CRNN with CTC loss.

    python -m src.crnn.train --config configs/crnn/full.yaml
    python -m src.crnn.train --config configs/crnn/full.yaml --resume      # after a Kaggle disconnect

Run from the repository root. A checkpoint is written after every epoch (last.pt) and whenever
the validation CER improves (best.pt). `max_minutes` stops training cleanly when the time budget
is used up, so a run always ends with a usable best.pt.

Where this file fits in the pipeline:
    prepare_iiit.py (CSVs + vocab) -> train.py (this file) -> checkpoints/<run>/best.pt
    -> evaluate.py (test the best checkpoint once).

What one training step does:
    batch of images -> CRNN -> log-probabilities (T, B, C) -> CTC loss against the true labels
    -> backward pass -> clip gradients -> AdamW step -> learning-rate schedule step.
After every epoch the model is scored on the validation set, and early stopping watches the val CER.
"""
# Standard library: command-line arguments, CSV log, file paths, timing.
import argparse
import csv
import os
import time

# torch + nn for the model and loss; yaml reads the config file.
import torch
import yaml
from torch import nn

# Shared code: metrics and the seed function.
from src.common.metrics import cer, wer
from src.common.seed import set_seed

# Code of this package: data, decoding and the network.
from .dataset import CharVocab, WordDataset, collate_fn
from .decode import greedy_decode
from .model import CRNN, count_params

# ===== CONCEPTS USED IN THIS FILE =====
# CRNN: the CNN + BiLSTM network from model.py that is trained here. It outputs one class score set per column.
# CTC loss (nn.CTCLoss): compares the per-column predictions with the true label without needing letter positions.
#    It sums over all valid alignments. Used as `ctc` in run() and evaluate_loader().
# Blank symbol: class 0, "no character here". CTCLoss(blank=CharVocab.BLANK) tells the loss which class it is.
# zero_infinity: if a label cannot fit in the time steps the loss would be inf; this turns it into 0 loss and
#    0 gradient so training does not break.
# Time steps and out_lens: columns after the CNN (width // 4). CTC gets the valid number per image, so padding is ignored.
# Greedy decoding: best class per step, merge repeats, drop blanks (decode.py). Used to get text for CER and WER.
# CER / WER: character and word error rate from metrics.py. CER on the validation set picks the best model.
# Epoch, batch: one epoch = one pass over all training data; a batch = the group of images used for one update.
# Training / validation set: training data changes the weights; validation data only measures progress.
# AdamW: an optimiser (Adam) with weight decay applied separately from the gradient step. Built as `opt`.
# Weight decay: a small push that shrinks weights, to reduce overfitting (weight_decay in the config).
# Learning rate and one-cycle schedule: lr rises to its peak over the first 15% of steps (pct_start=0.15), then
#    falls towards zero. OneCycleLR is stepped once per batch, so it needs total_steps.
# Gradient clipping: if the total gradient norm is above grad_clip, it is scaled down, so one bad batch cannot
#    cause a huge weight update (clip_grad_norm_).
# fp16 / mixed precision (autocast, GradScaler): the forward pass uses 16-bit numbers on a GPU for speed and
#    memory. GradScaler scales the loss so small fp16 gradients do not become zero. CTC loss stays float32.
# Seed: set_seed(cfg["seed"]) makes random choices repeatable (src/common/seed.py).
# Early stopping and patience: stop when val CER has not improved for `patience` epochs.
# Checkpoint (best.pt, last.pt): saved model files. best.pt = best val CER. last.pt also stores optimiser,
#    scheduler and scaler so --resume continues exactly.
# Resume: continue training from last.pt after an interruption (for example a Kaggle disconnect).
# Overfit test (overfit_n): train and test on a few samples; a correct model and loss must reach CER 0.
# model.train() / model.eval() / torch.no_grad(): dropout and BatchNorm behave differently in training and
#    evaluation; no_grad saves memory when no backward pass is needed.
# DataLoader / collate_fn / shuffle / pin_memory / num_workers: batching tools. See dataset.py.
# YAML config and argparse: settings in a YAML file, overridden by command-line flags (load_config, main).
# CSV log (log.csv): one row per epoch with loss, CER, WER, learning rate and time, used for the training curve.
# state_dict: the dictionary of a model's (or optimiser's) values, which is what torch.save stores.

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: Training of the CRNN-CTC word recogniser: it trains on the train CSV, validates each epoch and saves checkpoints.
# Q: How is it run?
# A: From the repo root: python -m src.crnn.train --config configs/crnn/full.yaml (add --resume to continue).
# Q: What does it read and write?
# A: It reads the config YAML, vocab.txt, the train and val CSVs with their images, and last.pt when resuming.
#    It writes best.pt, last.pt and log.csv into the checkpoint folder, and prints progress.
# Q: How is the best model chosen and when does training stop?
# A: The best model is the one with the lowest validation CER. Training stops at the last epoch, on early
#    stopping (patience), when max_minutes is used up, or when the overfit test reaches CER 0.

# Default settings. A YAML config file and command-line flags override these (see load_config).
DEFAULTS = dict(
    # Where the data is: folder with the CSVs and vocab.txt, the CSV names, and the folder that
    # the image paths inside the CSVs are relative to.
    data_dir="data/processed/words", train_csv="train.csv", val_csv="val.csv", image_root=".",
    # epochs = passes over the training data. lr = peak learning rate. weight_decay = AdamW decay.
    # grad_clip = maximum gradient norm. seed = random seed (see src/common/seed.py).
    epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4, grad_clip=5.0, num_workers=2, seed=42,
    # ckpt_dir = where checkpoints and log.csv go. overfit_n > 0 = sanity-check mode (see make_datasets).
    # max_minutes > 0 = stop after this many minutes (0 = no limit).
    ckpt_dir="checkpoints/crnn_full", run_name="crnn_full", overfit_n=0, max_minutes=0,
    # patience = epochs without val CER improvement before early stop. val_limit > 0 = use only the
    # first N validation samples (faster). invert = flip image colours. hidden, dropout = model settings.
    patience=8, val_limit=0, invert=False, hidden=256, dropout=0.3,
)


def load_config(path=None, **overrides):
    """Build the settings dict. Priority from low to high: DEFAULTS, YAML file, overrides.

    path: YAML file path or None. overrides: keyword values; a value of None means "not given" and is ignored,
    so command-line flags that the user did not set do not erase the YAML values.
    """
    # Q: What does it do and why does it exist?
    # A: It builds the one settings dict that run() uses, by layering DEFAULTS, then the YAML file, then the
    #    command-line overrides. This way every setting has a value and can be changed without editing code.
    # Q: Why dict(DEFAULTS) and why ignore None overrides?
    # A: dict() makes a copy, so DEFAULTS is not changed by update(). A flag the user did not type is None;
    #    if it were applied it would erase the value from the YAML file.
    cfg = dict(DEFAULTS) # copy, so DEFAULTS itself is never changed
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})  # "or {}" handles an empty YAML file
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def make_datasets(cfg, vocab):
    """Build the train and validation WordDatasets from the config. Returns (train, val)."""
    # Q: What does it do and why does it exist?
    # A: It creates the train and validation WordDatasets from the CSVs named in the config, and applies the
    #    overfit_n or val_limit option by cutting the sample lists.
    # Q: Why is val set to the train samples in overfit mode?
    # A: The overfit test checks that the model and loss can memorise a few samples. Scoring on the same
    #    samples shows whether it learned them. If CER cannot reach 0, there is a bug in the model or loss.
    d = cfg["data_dir"]
    train = WordDataset(os.path.join(d, cfg["train_csv"]), vocab, root=cfg["image_root"], invert=cfg["invert"])
    val = WordDataset(os.path.join(d, cfg["val_csv"]), vocab, root=cfg["image_root"], invert=cfg["invert"])
    if cfg["overfit_n"] > 0:  # sanity check: learn and test on the same few samples
        # If the model and loss are correct, it must be able to memorise a few samples (CER -> 0).
        train.samples = train.samples[: cfg["overfit_n"]]
        val.samples = train.samples
    elif cfg["val_limit"] > 0:
        val.samples = val.samples[: cfg["val_limit"]]
    return train, val


@torch.no_grad()  # no gradients are needed for evaluation: saves memory and time
def evaluate_loader(model, loader, vocab, device, ctc):
    """Run the model over a loader. Returns loss, CER, WER and the decoded texts.

    model: the CRNN. loader: DataLoader that uses collate_fn. vocab: CharVocab.
    device: torch device. ctc: an nn.CTCLoss instance.
    Returns a dict with loss (mean over batches), cer, wer, and the lists refs, hyps, paths
    (true texts, predicted texts, image paths). Also used by evaluate.py for the final test.
    """
    # Q: What does it do and why does it exist?
    # A: It runs the model over a whole DataLoader without training, decodes the predictions to text and returns
    #    the loss, CER and WER. train.py uses it after every epoch and evaluate.py uses it for the test set.
    # Q: Why model.eval(), @torch.no_grad() and skipping non-finite losses?
    # A: eval() turns off dropout and makes BatchNorm use stored statistics, so results are stable. no_grad saves
    #    memory because there is no backward pass. An inf or NaN loss would spoil the average, so it is skipped.
    model.eval()  # turns off dropout and makes BatchNorm use its stored running statistics
    total_loss, n_batches, refs, hyps, paths = 0.0, 0, [], [], []
    for batch in loader:
        images = batch["images"].to(device)  # (B, 1, 64, W)
        out_lens = CRNN.output_lengths(batch["widths"])  # (B,) valid time steps per image
        log_probs = model(images).float()  # (T, B, C); float32 for the loss
        loss = ctc(log_probs, batch["targets"], out_lens, batch["target_lengths"])
        if torch.isfinite(loss):  # skip a batch with an invalid loss so it does not ruin the average
            total_loss += float(loss)
            n_batches += 1
        # Decode on the CPU: the decoder is plain Python and works on lists.
        hyps += greedy_decode(log_probs.cpu(), out_lens, vocab)
        refs += batch["texts"]
        paths += batch["paths"]
    return {"loss": total_loss / max(n_batches, 1), "cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "refs": refs, "hyps": hyps, "paths": paths}


def run(cfg, resume=False):
    """Train the model as described by cfg. If resume=True, continue from last.pt when it exists.

    Returns a dict: best_cer (best validation CER), last_epoch (last finished epoch),
    train_time_min (total training minutes, including time before a resume).
    """
    # Q: What does it do and why does it exist?
    # A: It is the whole training procedure: set the seed, build data, model, AdamW, one-cycle schedule and CTC
    #    loss, then loop over epochs (train, validate, log, save checkpoints, check stop rules).
    # Q: Why is the loss computed outside autocast, and why is a resume refused when total_steps differs?
    # A: CTC sums many small probabilities, so it is computed in float32 for accuracy. The saved one-cycle
    #    schedule was built for a fixed number of steps, so a different epoch count or data size would put
    #    the learning rate on the wrong point of the curve.
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision (fp16) only works on a GPU here
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    # ---- data
    vocab = CharVocab.from_file(os.path.join(cfg["data_dir"], "vocab.txt"))
    train_ds, val_ds = make_datasets(cfg, vocab)
    # shuffle=True for training so batches differ each epoch; the val set keeps a fixed order.
    # pin_memory speeds up the copy of batches to the GPU.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        collate_fn=collate_fn, pin_memory=use_amp)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
        collate_fn=collate_fn, pin_memory=use_amp)

    # ---- model, optimiser, schedule, loss
    model = CRNN(len(vocab), hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    # AdamW = Adam with weight decay applied separately from the gradient update (decoupled decay).
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # The scheduler is stepped once per batch, so it needs the total number of batches in the whole run.
    total_steps = max(cfg["epochs"] * len(train_loader), 1)
    # One-cycle schedule: the learning rate rises to max_lr during the first 15% of steps (pct_start),
    # then falls smoothly towards zero until the end.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total_steps, pct_start=0.15)
    # GradScaler multiplies the loss before backward so small fp16 gradients do not become zero.
    # With enabled=False (CPU) it does nothing.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # blank=0 matches CharVocab.BLANK. zero_infinity=True turns an infinite loss (label longer than
    # the available time steps) into 0 with 0 gradient, instead of breaking training with inf / NaN.
    ctc = nn.CTCLoss(blank=CharVocab.BLANK, zero_infinity=True)

    # ---- optional resume from the last checkpoint
    start_epoch, best_cer, bad_epochs, minutes_before = 0, float("inf"), 0, 0.0
    last_path = os.path.join(cfg["ckpt_dir"], "last.pt")
    if resume and os.path.exists(last_path):
        # weights_only=False: the file holds plain Python objects (cfg, chars) as well as tensors.
        ck = torch.load(last_path, map_location=device, weights_only=False)
        # The saved schedule only makes sense for the same number of steps, so refuse a mismatch.
        if ck["sched"]["total_steps"] != total_steps:
            raise ValueError("resume needs the same epochs and data as the original run: the learning-rate "
                             "schedule was built for a different number of steps")
        # Restore everything that changes during training, so the run continues as if never stopped.
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_cer = ck["epoch"] + 1, ck["best_cer"]
        bad_epochs, minutes_before = ck["bad_epochs"], ck["train_time_min"]
        print(f"resumed from epoch {ck['epoch']} (best val CER so far {best_cer:.4f})", flush=True)

    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  classes={len(vocab)}  "
          f"params={count_params(model):,}", flush=True)
    # log.csv gets one row per epoch (used by scripts/plot_crnn_curve.py). A resumed run appends to it.
    log_path = os.path.join(cfg["ckpt_dir"], "log.csv")
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_cer", "val_wer", "lr", "minutes"])

    # ---- training loop
    t_start = time.time()
    # Total training minutes = minutes from before a resume + time since this process started.
    minutes = lambda: minutes_before + (time.time() - t_start) / 60  # noqa: E731
    time_up = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()  # dropout on, BatchNorm updates its statistics
        run_loss, n_ok = 0.0, 0  # sum of batch losses and number of batches that were used
        for batch in train_loader:
            images = batch["images"].to(device)  # (B, 1, 64, W)
            out_lens = CRNN.output_lengths(batch["widths"])  # (B,) valid time steps per image
            # Forward pass in fp16 on GPU: faster and uses less memory. (No-op on CPU.)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                log_probs = model(images)  # (T, B, C)
            # The loss is computed outside autocast in float32: CTC sums many small probabilities,
            # so it needs full precision.
            loss = ctc(log_probs.float(), batch["targets"], out_lens, batch["target_lengths"])
            if not torch.isfinite(loss):  # bad batch (inf / NaN): throw it away, do not update the weights
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)  # clear old gradients (set_to_none is a little faster)
            scaler.scale(loss).backward()  # backward pass on the scaled loss
            scaler.unscale_(opt)  # undo the scaling first, so the clipping below sees the true gradient size
            # Gradient clipping: if the total gradient norm is above grad_clip, shrink it.
            # This stops one bad batch from making a huge, harmful weight update.
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)  # optimiser step (skipped by the scaler if gradients contain inf / NaN)
            scaler.update()  # adjust the scale factor for the next step
            sched.step()  # move the learning rate along the one-cycle schedule (once per batch)
            run_loss += loss.item()
            n_ok += 1
            if cfg["max_minutes"] and minutes() >= cfg["max_minutes"]:
                time_up = True  # stop the epoch now; after validation and saving, the run ends
                break
        train_loss = run_loss / max(n_ok, 1)

        # ---- validation after every epoch
        res = evaluate_loader(model, val_loader, vocab, device, ctc)
        # Model selection uses CER (lower is better): it is the metric we report, and it is not
        # the same thing as the loss.
        improved = res["cer"] < best_cer
        if improved:
            best_cer, bad_epochs = res["cer"], 0
        else:
            bad_epochs += 1  # counts epochs without improvement, for early stopping
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | train loss {train_loss:7.3f} | val loss {res['loss']:7.3f} | "
              f"val CER {res['cer']:.4f} | val WER {res['wer']:.4f} | lr {lr_now:.2e} | {minutes():.1f} min", flush=True)
        # Show 3 examples so a human can see what the model predicts.
        for r, h in list(zip(res["refs"], res["hyps"]))[:3]:
            print(f"      truth: {r}   predicted: {h}", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{res['loss']:.4f}", f"{res['cer']:.4f}",
                                    f"{res['wer']:.4f}", f"{lr_now:.3e}", f"{minutes():.2f}"])

        # ---- checkpoints
        # best.pt: only what is needed to use the model later. chars lets evaluate.py rebuild the vocab.
        state = {"model": model.state_dict(), "chars": vocab.chars, "cfg": cfg, "epoch": epoch,
                 "val_cer": res["cer"], "train_time_min": minutes()}
        if improved:
            torch.save(state, os.path.join(cfg["ckpt_dir"], "best.pt"))
        # last.pt: the same plus optimiser, scheduler and scaler state, so --resume can continue exactly.
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_cer": best_cer, "bad_epochs": bad_epochs}, last_path)
        last_epoch = epoch

        # ---- stop rules (checked after saving, so the checkpoints are always up to date)
        if cfg["overfit_n"] > 0 and res["cer"] == 0.0:
            print("overfit test passed: CER is 0 on the training samples", flush=True)
            break
        if time_up:
            print(f"time budget of {cfg['max_minutes']} min used up: stopping", flush=True)
            break
        if bad_epochs >= cfg["patience"]:
            print(f"no improvement for {cfg['patience']} epochs: early stop", flush=True)
            break

    return {"best_cer": best_cer, "last_epoch": last_epoch, "train_time_min": minutes()}


def main():
    """Command-line entry point: read flags, build the config and call run()."""
    # Q: What does it do and why does it exist?
    # A: It is what runs when you type python -m src.crnn.train. It reads the command-line flags, builds the
    #    config with load_config and starts run(), then prints the result dict.
    # Q: Why do the optional flags have no default value?
    # A: Their default is None, which load_config ignores. So only the flags you really type override the
    #    YAML file, and the other settings stay as in the config.
    ap =argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # YAML file with the base settings
    ap.add_argument("--resume", action="store_true")
    # The flags below are optional. Any flag that is left out is None and does not override the config.
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max_minutes", type=float)
    ap.add_argument("--overfit_n", type=int)
    ap.add_argument("--data_dir")
    ap.add_argument("--image_root")
    ap.add_argument("--ckpt_dir")
    ap.add_argument("--num_workers", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config, epochs=a.epochs, max_minutes=a.max_minutes, overfit_n=a.overfit_n,
                      data_dir=a.data_dir, image_root=a.image_root, ckpt_dir=a.ckpt_dir,
                      num_workers=a.num_workers)
    print(run(cfg, resume=a.resume))


if __name__ == "__main__":
    main()
