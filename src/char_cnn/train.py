"""Train the character CNN (Model A) on DHCD.

    python -m src.char_cnn.train --config configs/char_cnn/full.yaml
    python -m src.char_cnn.train --config configs/char_cnn/full.yaml --resume   # after a disconnect

Run from the repository root. 10 % of the official Train folder (stratified, seed 42) is the
validation set; the Test folder is not touched here (src.char_cnn.evaluate uses it once).
A checkpoint is written after every epoch (last.pt) and whenever validation accuracy improves
(best.pt). `max_minutes` stops training cleanly when the time budget is used up.

Where this file fits in the pipeline:
    dataset.py (data) + model.py (network) -> train.py (this file) -> checkpoints/.../best.pt
    -> evaluate.py (final test) and predict.py (use on new images).
    One epoch = one full pass over the training images, in mini-batches.
"""
# argparse: command-line options. csv: the training log. os: paths. time: the time budget.
import argparse
import csv
import os
import time

# numpy: random sampling for the overfit test. torch: training. yaml: read the config file.
# nn: the loss function.
import numpy as np
import torch
import yaml
from torch import nn

# set_seed fixes all random generators (Python, NumPy, torch) so runs can be repeated.
from src.common.seed import set_seed

from .dataset import CHARS, NUM_CLASSES, load_split, random_affine, stratified_split, to_input
from .model import CharCNN, count_params

# CONCEPTS USED IN THIS FILE
# - Epoch: one full pass over all training images. The outer loop in run goes over epochs.
# - Mini-batch / batch size: the images are processed in small groups (128) and the weights are updated after each
#   group. The inner loop in run. No DataLoader is used: the data sits on the device and a random permutation
#   (randperm) picks each batch.
# - Cross-entropy loss: the error measure for classification (nn.CrossEntropyLoss); it is small when the true class
#   gets a high score. train_loss_fn and eval_loss_fn.
# - Label smoothing: the target is not 100 % on the true class but a little spread over the others, so the model is
#   not over-confident. Used in the training loss only.
# - Backpropagation / gradient: loss.backward() computes how each weight should change; the optimizer then updates the
#   weights. zero_grad clears old gradients first.
# - Adam: an optimizer that adapts the step size of each weight. lr is the learning rate (step size); weight_decay
#   adds a small penalty on large weights.
# - Cosine schedule (CosineAnnealingLR): the learning rate falls smoothly from lr towards 0 over the epochs.
#   sched.step() is called once per epoch.
# - Mixed precision (autocast + GradScaler): the forward pass uses float16 on a GPU for speed and memory; GradScaler
#   scales the loss so tiny float16 gradients do not become 0. Only when use_amp is True.
# - Data augmentation: random_affine changes each training batch a little (rotate, shift, scale, shear). Switched off
#   for validation and in the overfit test.
# - Validation vs test: a stratified 10 % of Train is the validation set used to pick best.pt and stop early; the Test
#   split is not loaded here (make_data).
# - Seed (set_seed): fixes the random generators so a run can be repeated. Called first in run.
# - train() / eval() mode and no_grad: train mode has dropout on and BatchNorm updating; eval mode has dropout off.
#   @torch.no_grad() in evaluate stores no gradients, saving memory.
# - Checkpoint and resume: best.pt (best validation accuracy) and last.pt (latest epoch plus optimizer, scheduler and
#   scaler state) let training continue after a disconnect (--resume).
# - Early stopping and patience: stop if validation accuracy has not improved for `patience` epochs. max_minutes is a
#   separate time budget.
# - Overfit test (overfit_n): train and validate on a few samples; a correct model and loop must reach 100 %,
#   otherwise there is a bug.
# - Accuracy: share of images whose argmax class equals the label.
# - YAML config + argparse: settings come from DEFAULTS, then the YAML file, then command-line overrides (load_config,
#   main).

# FILE Q&A (viva)
# Q: What part of the project is this file?
# A: Training of Model A: it loads DHCD Train, splits off validation, trains CharCNN and saves checkpoints and a log.
# Q: How is it run?
# A: From the repo root: python -m src.char_cnn.train --config configs/char_cnn/full.yaml. Add --resume to continue
#    from last.pt. Optional overrides: --epochs, --max_minutes, --overfit_n, --data_dir, --ckpt_dir.
# Q: What does it read and write?
# A: It reads the YAML config and the DHCD Train split (via load_split and its cache). In ckpt_dir it writes log.csv,
#    best.pt and last.pt. The Test split is never read.
# Q: When does training stop?
# A: When all epochs are done, when validation accuracy has not improved for `patience` epochs, when max_minutes is
#    used up, or when the overfit test reaches accuracy 1.0.

# Default settings. A YAML config file or command-line flags override them (see load_config).
# val_fraction: share of Train kept for validation. lr: learning rate. weight_decay: L2-style penalty
# that keeps weights small. label_smoothing: softens the one-hot targets. augment: random affine on/off.
# overfit_n: if > 0, run the sanity test on that many samples. max_minutes: 0 = no time limit.
# patience: stop after this many epochs without a better validation accuracy.
DEFAULTS = dict(
    data_dir="data/raw/dhcd", val_fraction=0.1, epochs=30, batch_size=128, lr=1e-3, weight_decay=1e-4,
    label_smoothing=0.1, dropout=0.4, augment=True, seed=42, ckpt_dir="checkpoints/char_cnn_full",
    run_name="char_cnn_full", overfit_n=0, max_minutes=0, patience=8,
)


def load_config(path=None, **overrides):
    """Build the settings dict. Priority (low to high): DEFAULTS, YAML file, overrides.

    path: YAML file path or None. overrides: keyword settings; a value of None is ignored,
    so unset command-line flags do not erase the YAML value. Returns a new dict.
    # Q: What does load_config do and why does it exist?
    # A: It builds the settings dict from DEFAULTS, then the YAML file, then keyword overrides. One dict then drives
    #    the whole run.
    # Q: Why copy DEFAULTS and ignore None overrides?
    # A: dict(DEFAULTS) keeps DEFAULTS unchanged. Command-line flags that were not given are None, so ignoring None
    #    keeps the YAML value instead of erasing it.
    """
    cfg = dict(DEFAULTS)  # copy, so DEFAULTS itself is never changed
    if path:
        with open(path, encoding="utf-8") as f:
            # `or {}` handles an empty YAML file (safe_load returns None).
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def make_data(cfg):
    """Return (x_train, y_train, x_val, y_val) as uint8 images and int64 labels (CPU tensors).

    Loads the official Train split and cuts a stratified validation part out of it.
    The official Test split is not loaded here, so it cannot leak into model choice.
    """
    # Q: What does make_data do and why does it exist?
    # A: It loads the official Train split and cuts a stratified validation part out of it, returning train and
    #    validation images and labels. The Test split is not loaded here.
    # Q: Why does overfit_n make the validation set equal to the training samples?
    # A: It is a sanity check: a correct model and loop must memorise a few samples. Validating on the same samples
    #    should reach 100 %; if not, there is a bug somewhere.
    images, labels, _ = load_split(cfg["data_dir"], "Train")
    # Same seed in the config -> same split on every run (and after --resume).
    tr, va = stratified_split(labels.numpy(), cfg["val_fraction"], cfg["seed"])
    if cfg["overfit_n"] > 0:  # sanity check: learn and test on the same few samples
        # Idea: a correct model and training loop must be able to memorise a handful of samples.
        # If it cannot reach 100 %, there is a bug somewhere.
        rng = np.random.default_rng(cfg["seed"])
        tr = np.sort(rng.choice(tr, size=min(cfg["overfit_n"], len(tr)), replace=False))
        va = tr  # validate on the same samples
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)
    # Index with the tensors: images (N,32,32) -> (n_train,32,32) and (n_val,32,32).
    return images[tr], labels[tr], images[va], labels[va]


@torch.no_grad()
def evaluate(model, x_uint8, y, device, criterion, batch_size=1024):
    """Loss, accuracy and predicted class indices for a whole set.

    model: CharCNN. x_uint8: images (N,32,32) uint8 (CPU). y: labels (N,) int64 (CPU).
    device: where to run. criterion: loss function. batch_size: images per step (large is fine
    because no gradients are stored).
    Returns dict {"loss": float, "acc": float 0..1, "preds": tensor (N,) of class indices on CPU}.
    No augmentation here, so the numbers measure the real images.
    Also imported by evaluate.py, so validation and test use the same code.
    """
    # Q: What does evaluate do and why does it exist?
    # A: It runs the model over a whole set and returns loss, accuracy and predicted classes. It is used for
    #    validation each epoch and imported by evaluate.py for the test.
    # Q: Why model.eval(), no_grad, and loss times batch size?
    # A: eval mode turns dropout off and makes BatchNorm use its saved statistics, so results are stable. no_grad
    #    saves memory. The criterion returns a batch mean, so multiplying by the batch size and dividing by N at the
    #    end weights a smaller last batch correctly.
    # eval mode: dropout off, BatchNorm uses its running statistics. @torch.no_grad() saves memory and time.
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(y), batch_size):
        # Move one batch at a time to the device (the full set may not fit on the GPU).
        xb = to_input(x_uint8[i:i + batch_size].to(device))
        yb = y[i:i + batch_size].to(device)
        logits = model(xb).float()  # (B, 46); float32 for a precise loss
        # criterion returns the batch mean; multiply by the batch size to get a sum, then average over N at the end.
        total_loss += float(criterion(logits, yb)) * len(yb)
        preds.append(logits.argmax(1).cpu())  # index of the highest score = predicted class
    # Handle an empty set without crashing.
    preds = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    acc = float((preds == y).float().mean()) if len(y) else 0.0
    return {"loss": total_loss / max(len(y), 1), "acc": acc, "preds": preds}


def run(cfg, resume=False):
    """Train the model with settings `cfg` and return a small summary dict.

    resume: if True and last.pt exists, continue from it (for example after a Kaggle disconnect).
    Returns {"best_val_acc", "last_epoch", "train_time_min"}.
    """
    # Q: What does run do and why does it exist?
    # A: It is the whole training procedure: set up data, model, optimizer, scheduler and losses, loop over epochs
    #    with mini-batches, validate, log, save checkpoints, apply the stop rules. It returns a summary.
    # Q: Why must resume use the same epochs, and why does last.pt store optimizer state?
    # A: The cosine schedule depends on epochs (T_max), so the code raises ValueError if they differ. last.pt also
    #    stores optimizer, scheduler and scaler state so training continues exactly where it stopped.
    # Seed first, so weight init, shuffling and augmentation are repeatable.
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mixed precision (float16) is used only on a GPU: it is faster and uses less memory there.
    use_amp = device.type == "cuda"
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    x_tr, y_tr, x_va, y_va = make_data(cfg)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)  # about 70 MB: fits easily on the GPU
    # Keeping the training set on the GPU avoids a CPU -> GPU copy at every step.

    model = CharCNN(NUM_CLASSES, dropout=cfg["dropout"]).to(device)
    # Adam: adaptive learning rate per weight, a safe default choice. weight_decay adds a small penalty on large weights.
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # Cosine schedule: the learning rate goes smoothly from lr down towards 0 over `epochs` epochs.
    # Big steps first, small careful steps at the end.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    # GradScaler multiplies the loss before backward so tiny float16 gradients do not become 0.
    # It does nothing when use_amp is False.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Label smoothing: the target is 90 % on the true class and the rest spread over the others,
    # so the model does not become over-confident. Used for training only.
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    # Plain loss for validation/test, so val loss is the true cross-entropy and easy to compare.
    eval_loss_fn = nn.CrossEntropyLoss()

    # State that must survive a restart: epoch to start at, best val accuracy so far,
    # epochs without improvement (for early stopping), and training minutes already used.
    start_epoch, best_acc, bad_epochs, minutes_before = 0, -1.0, 0, 0.0
    last_path = os.path.join(cfg["ckpt_dir"], "last.pt")
    if resume and os.path.exists(last_path):
        # weights_only=False because the checkpoint also holds plain Python objects (cfg dict, chars list).
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["cfg"]["epochs"] != cfg["epochs"]:
            raise ValueError("resume needs the same 'epochs' as the original run (the cosine schedule depends on it)")
        # Restore the model weights and the optimizer / scheduler / scaler state,
        # so training continues exactly where it stopped.
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]  # continue with the NEXT epoch
        bad_epochs, minutes_before = ck["bad_epochs"], ck["train_time_min"]
        print(f"resumed from epoch {ck['epoch']} (best val accuracy so far {best_acc:.4f})", flush=True)

    # flush=True so the line shows at once in Kaggle / notebook logs.
    print(f"device={device}  train={len(y_tr)}  val={len(y_va)}  classes={NUM_CLASSES}  "
          f"params={count_params(model):,}", flush=True)
    log_path = os.path.join(cfg["ckpt_dir"], "log.csv")
    # New run: start a fresh log with a header. Resume: keep the old log and append to it.
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "minutes"])

    t_start = time.time()
    # Total training minutes = minutes from earlier runs (resume) + minutes since this run started.
    minutes = lambda: minutes_before + (time.time() - t_start) / 60  # noqa: E731
    # No augmentation in the overfit test: we want the model to memorise the exact samples.
    augment = cfg["augment"] and cfg["overfit_n"] == 0
    time_up, last_epoch = False, start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()  # train mode: dropout on, BatchNorm updates its statistics
        # New random order of the training images every epoch (done on the device).
        perm = torch.randperm(len(y_tr), device=device)
        run_loss, correct, seen = 0.0, 0, 0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]  # indices of this mini-batch
            xb, yb = to_input(x_tr[idx]), y_tr[idx]  # xb (B,1,32,32) float, yb (B,) int64
            if augment:
                xb = random_affine(xb)  # random rotate / shift / scale / shear, same shape
            # Forward pass in float16 where it is safe (GPU only).
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)  # (B, 46)
            # Loss in float32 for accuracy.
            loss = train_loss_fn(logits.float(), yb)
            # set_to_none=True frees the old gradients (a little faster than filling them with 0).
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # backpropagation on the scaled loss
            scaler.step(opt)  # unscale the gradients and update the weights (skips the step on inf/NaN)
            scaler.update()  # adjust the scale factor for the next step
            # Running totals for the epoch's average loss and accuracy (loss is a batch mean, so weight by batch size).
            run_loss += loss.item() * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(yb)
            # Time budget check inside the epoch, so we stop soon after the limit.
            if cfg["max_minutes"] and minutes() >= cfg["max_minutes"]:
                time_up = True
                break
        sched.step()  # move the learning rate along the cosine curve (once per epoch)
        # max(seen, 1) avoids division by zero. These are the average loss / accuracy over the epoch.
        train_loss, train_acc = run_loss / max(seen, 1), correct / max(seen, 1)

        # Validate after every epoch on data the model never trains on.
        res = evaluate(model, x_va, y_va, device, eval_loss_fn)
        improved = res["acc"] > best_acc
        if improved:
            best_acc, bad_epochs = res["acc"], 0  # new best: reset the early-stopping counter
        else:
            bad_epochs += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {res['loss']:.4f} acc {res['acc']:.4f} | lr {lr_now:.2e} | {minutes():.1f} min", flush=True)
        # One row per epoch; evaluate.py reads this file to draw the learning curve.
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{res['loss']:.4f}",
                                    f"{res['acc']:.4f}", f"{lr_now:.3e}", f"{minutes():.2f}"])

        # `state` holds what is needed to use the model later (weights, class characters, config).
        state = {"model": model.state_dict(), "chars": CHARS, "cfg": cfg, "epoch": epoch,
                 "val_acc": res["acc"], "train_time_min": minutes()}
        if improved:
            # best.pt = the epoch with the best validation accuracy (used for the final test).
            torch.save(state, os.path.join(cfg["ckpt_dir"], "best.pt"))
        # last.pt = latest epoch plus the optimizer/scheduler/scaler state, so --resume can continue.
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_acc": best_acc, "bad_epochs": bad_epochs}, last_path)
        last_epoch = epoch

        # Stop rules, checked after the checkpoint is saved so no work is lost.
        if cfg["overfit_n"] > 0 and res["acc"] == 1.0:
            print("overfit test passed: accuracy is 1.0 on the training samples", flush=True)
            break
        if time_up:
            print(f"time budget of {cfg['max_minutes']} min used up: stopping", flush=True)
            break
        # Early stopping: no better validation accuracy for `patience` epochs, so more training would likely not help.
        if bad_epochs >= cfg["patience"]:
            print(f"no improvement for {cfg['patience']} epochs: early stop", flush=True)
            break

    return {"best_val_acc": best_acc, "last_epoch": last_epoch, "train_time_min": minutes()}


def main():
    """Command-line entry point: read the flags, build the config, train, print the summary."""
    # Q: What does main do and why does it exist?
    # A: It is the command-line entry point: it reads the flags, builds the config with load_config, calls run and
    #    prints the summary.
    # Q: Why are most flags optional with no default?
    # A: Their default is None, and load_config ignores None, so the YAML file (or DEFAULTS) supplies the value unless
    #    a flag is given.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # path to the YAML file
    ap.add_argument("--resume", action="store_true")  # flag: continue from last.pt
    # The next flags are optional overrides of the YAML values (None when not given).
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max_minutes", type=float)
    ap.add_argument("--overfit_n", type=int)
    ap.add_argument("--data_dir")
    ap.add_argument("--ckpt_dir")
    a = ap.parse_args()
    cfg = load_config(a.config, epochs=a.epochs, max_minutes=a.max_minutes, overfit_n=a.overfit_n,
                      data_dir=a.data_dir, ckpt_dir=a.ckpt_dir)
    print(run(cfg, resume=a.resume))


if __name__ == "__main__":
    main()
