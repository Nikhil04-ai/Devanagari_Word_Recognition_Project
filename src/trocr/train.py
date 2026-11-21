"""Train the TrOCR-style model (plain PyTorch loop, no Trainer).

    python -m src.trocr.train --config configs/trocr/overfit.yaml     # sanity check: memorise 100 images
    python -m src.trocr.train --config configs/trocr/full.yaml        # full run, stops at its time budget
    python -m src.trocr.train --config configs/trocr/full.yaml --resume   # after a Kaggle disconnect

Run from the repository root. A checkpoint is written after every epoch (last.pt) and whenever the
validation CER improves (best.pt). `max_minutes` stops training cleanly when the time budget is used
up, so a run always ends with a usable best.pt.

Where it fits in the pipeline: config.py -> dataset.py (batches) + model.py (network) -> THIS FILE
(learning) -> checkpoints/.../best.pt -> evaluate.py (final test score).

What one training step does:
    1. forward pass with teacher forcing -> loss (cross-entropy over characters)
    2. backward pass -> gradients (added up over `grad_accum` mini-batches)
    3. unscale + clip gradients -> AdamW step -> learning-rate scheduler step
After every epoch the model is scored on the validation set (CER = character error rate,
WER = word error rate; lower is better) and checkpoints are saved.
"""
# Standard library: argparse (command line), csv (training log), math (cosine), os (paths), time (time budget).
import argparse
import csv
import math
import os
import time

# torch for the model and the optimizer; LambdaLR is a scheduler driven by our own function.
import torch
from torch.optim.lr_scheduler import LambdaLR

# Shared project code: the CER / WER metrics and the function that fixes all random seeds.
from src.common.metrics import cer, wer
from src.common.seed import set_seed

# Local modules: config, data loaders, and model helpers.
from .config import load_config, parse_overrides, to_dict
from .dataset import get_loaders
from .model import build_model, count_params, generate_ids, new_param_names, set_encoder_trainable

# ======================================= CONCEPTS USED IN THIS FILE =======================================
# TrOCR: Transformer-based OCR from Microsoft: an image encoder plus a text decoder. This file trains it on
#     IIIT-HW-Dev word images, starting from pre-trained weights (build_model in run).
# Encoder-decoder: the encoder turns the image into features, the decoder writes the word from them, one
#     character at a time (see model.py). Fine-tuning here means continuing to train both parts on our data.
# Teacher forcing: while training, the decoder is given the TRUE previous characters, not its own guesses.
#     model(pixel_values=px, labels=labels) in run and in evaluate_loader (with_loss) does this.
# Cross-entropy loss and -100 labels: the loss compares the predicted character scores with the true character.
#     Padding positions are labelled -100 and are ignored. The model returns it as `.loss`.
# Greedy generation: at test time the model writes the word itself, taking the most likely character each step
#     (generate_ids with num_beams = 1). Used to score the validation set in evaluate_loader.
# CER (character error rate): edit distance divided by number of true characters; lower is better. WER (word error
#     rate): same, counted over words. Both come from src/common/metrics.py and are computed in evaluate_loader.
# AdamW: the optimizer that updates the weights from their gradients; weight_decay is a small pull of weights
#     towards zero (cfg.weight_decay). Built in make_optimizer.
# Two learning rates: pre-trained weights get a small lr (lr_pretrained), new random weights get a larger one
#     (lr_new). Done with two parameter groups in make_optimizer.
# Learning-rate scheduler (LambdaLR): changes the learning rate during training with our own function f.
# Warm-up: the lr rises linearly from almost 0 to its full value in the first steps (make_scheduler). It is capped
#     at 10% of all steps in run.
# Cosine decay: after warm-up the lr falls along a cosine curve to 0, so the last steps are small (make_scheduler).
# Epoch, batch, step: an epoch is one pass over the training data; a batch is a group of images; a (optimizer) step
#     is one weight update. The lr schedule counts optimizer steps.
# Gradient accumulation: gradients of cfg.grad_accum batches are added up before one weight update, which acts
#     like a bigger batch. The loss is divided by grad_accum so the sum is an average (see run).
# fp16 / mixed precision / autocast: half-precision math on the GPU, faster and lighter. torch.autocast switches
#     it on; `amp` is True only on a GPU.
# GradScaler: multiplies the loss by a large number before backward so tiny fp16 gradients do not become zero
#     (underflow). unscale_() undoes it, step() and update() apply it. Does nothing when amp is off.
# Gradient clipping: if the total gradient norm is above cfg.grad_clip, all gradients are scaled down. This guards
#     against exploding gradients (clip_grad_norm_ in run).
# NaN / inf loss skip: a batch with a non-finite loss is skipped (torch.isfinite) so it cannot damage the weights.
# Freezing the encoder: requires_grad = False stops it from learning (set_encoder_trainable). cfg.freeze_encoder_epochs
#     sets how many first epochs are decoder-only; model.encoder.eval() also keeps its dropout off.
# train() / eval() and dropout: train() switches dropout (random switching-off of units) on; eval() switches it off so
#     validation is repeatable. @torch.no_grad() stops gradient tracking when only predicting.
# Validation set: images not used for learning; scored after every epoch to see real progress and choose the model.
# Checkpoint: a saved file with the weights and more. best.pt = weights of the best epoch; last.pt = latest epoch plus
#     optimizer, scheduler and scaler state.
# Best-epoch selection: the epoch with the lowest validation CER is saved as best.pt (variables improved, best_cer).
# Early stopping / patience: stop if validation CER did not improve for cfg.patience epochs in a row (`bad`).
# Time budget (max_minutes): training stops cleanly when this many minutes are used; minutes() counts them.
# Resume: --resume reloads last.pt (weights, optimizer, scheduler, scaler, counters) and continues from the next epoch.
# Overfit test: with overfit_n > 0 the model trains on a few images only; reaching CER 0 shows the code can learn.
# Seed: set_seed(cfg.seed) fixes the random number generators so a run can be repeated.
# log.csv: one row per epoch (losses, CER, WER, lr, minutes), used later to plot training curves.
# set_to_none=True in zero_grad: frees the gradient memory instead of filling it with zeros.
# non_blocking=True: lets the copy of data to the GPU overlap with other work.
# argparse / overrides: command-line options replace the matching keys of the YAML config (main).
# ==========================================================================================================

# ======================================= FILE-LEVEL Q&A (VIVA) ============================================
# Q: What does this file do?
# A: It trains the TrOCR-style model with a plain PyTorch loop. After every epoch it scores the validation set
#    with CER / WER, writes a log line and saves checkpoints. evaluate.py later tests best.pt on the test set.
# Q: How is it run?
# A: From the repository root: python -m src.trocr.train --config configs/trocr/full.yaml
#    Use configs/trocr/overfit.yaml for the sanity check and add --resume to continue after a disconnect.
# Q: What does it read and what does it write?
# A: It reads the YAML config (plus command-line overrides) and the train / val CSV files and images through
#    get_loaders. It writes best.pt, last.pt and log.csv into cfg.ckpt_dir and prints progress to the screen.
# Q: Why is evaluate_loader in this file and also used by evaluate.py?
# A: Validation after each epoch and the final test need the same job (generate words, compute CER and WER), so
#    it is written once here and imported by evaluate.py.
# ==========================================================================================================


def make_optimizer(model, cfg):
    """AdamW with two parameter groups: new embeddings/head (lr_new) vs pre-trained (lr_pretrained).

    Input: model, cfg. Returns: a torch AdamW optimizer.
    Why two learning rates: the pre-trained weights are already good, so a big step could damage
    them; the new random weights start from nothing and need a bigger step to learn quickly."""
    # Q: What does it do and why does it exist?
    # A: It builds the AdamW optimizer with two groups of parameters, each with its own learning rate. run calls it
    #    once before training starts.
    # Q: Why are frozen parameters also put in the groups, and why are empty groups removed?
    # A: The optimizer is built once, but set_encoder_trainable changes requires_grad later; a frozen parameter just
    #    gets no gradient and is not updated. AdamW raises an error for a group with no parameters.
    # Names of the randomly initialised (new) parameters; everything else is pre-trained.
    new_names = new_param_names(model, cfg)
    new, old = [], []
    for n, p in model.named_parameters():
        (new if n in new_names else old).append(p)  # frozen params simply get no gradient
    # Group 0 = pre-trained (small lr), group 1 = new (large lr). The order matters: train.py reads the last group's lr for logging.
    groups = [{"params": old, "lr": cfg.lr_pretrained}, {"params": new, "lr": cfg.lr_new}]
    # An empty group would raise an error, so it is filtered out.
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg.weight_decay)


def make_scheduler(opt, warmup, total):
    """Linear warm-up, then cosine decay to 0 (steps counted in optimizer updates).

    Inputs: optimizer, `warmup` = number of warm-up steps, `total` = total optimizer steps of the run.
    Returns: LambdaLR scheduler. At each step the base lr of every group is multiplied by f(step), a factor in (0, 1].
    Why warm-up: at the start the new layers are random and gradients are noisy, so a gentle start
    makes training more stable. Why cosine decay: the lr falls smoothly, so the last steps are small
    and fine-tune the weights."""
    # Q: What does it do and why does it exist?
    # A: It makes the learning-rate schedule: a short linear warm-up, then a cosine fall to 0. run creates it
    #    and calls sched.step() after every optimizer update.
    # Q: Why are `warmup` and `total` counted in optimizer steps and not in batches?
    # A: sched.step() is called once per optimizer update, which happens every grad_accum batches. If the numbers
    #    were in batches, the schedule would reach its end grad_accum times too early.
    def f(step):
        # Q: What does it do and why does it exist?
        # A: It returns the factor (0 to 1) that LambdaLR multiplies with each group's base learning rate at this
        #    step. It is the whole warm-up plus cosine shape of the schedule.
        # Q: Why (step + 1), max(1, ...) and min(1.0, prog)?
        # A: (step + 1) makes the very first step use a small non-zero lr instead of exactly 0. max(1, ...) avoids
        #    division by zero when warmup or the decay phase has length 0. min(1.0, prog) stops the cosine from
        #    going up again if the step count goes beyond `total`.
        # Warm-up: factor grows linearly from 1/warmup up to 1.
        if step < warmup:
            return (step + 1) / max(1, warmup)
        # Progress through the decay phase, from 0 to 1 (clamped at 1 so the factor never goes up again).
        prog = (step - warmup) / max(1, total - warmup)
        # Cosine curve: 1 at prog=0, 0 at prog=1.
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return LambdaLR(opt, f)


@torch.no_grad()
def evaluate_loader(model, loader, tok, cfg, device, amp, num_beams=1, with_loss=True):
    """Generate text for every batch. Returns a dict: loss, cer, wer, refs, hyps, paths.

    Inputs: model; loader = DataLoader; tok = tokenizer; cfg; device; amp = use fp16 autocast;
    num_beams = 1 for greedy decoding, >1 for beam search; with_loss = also compute the teacher-forced loss.
    Returns: loss (mean over batches), cer, wer, and lists refs (true words), hyps (predicted words), paths.
    Why greedy by default: it is fast, so it is used for validation after each epoch.
    @torch.no_grad(): no gradients are needed, which saves memory and time."""
    # Q: What does it do and why does it exist?
    # A: It runs the model over a whole DataLoader without learning: it generates a word for every image, then
    #    returns CER, WER and the texts. run uses it for validation after each epoch; evaluate.py uses it for the test.
    # Q: Why is the loss computed by teacher forcing but the words by generation?
    # A: The teacher-forced loss shows how well the model predicts each character when it is given the true
    #    previous ones. The generated words are what the model really writes alone, so CER and WER come from them.
    #    Skipping non-finite losses stops one bad batch from ruining the average loss.
    # eval() switches dropout off so results are repeatable.
    model.eval()
    refs, hyps, paths, tot_loss, n_batches = [], [], [], 0.0, 0
    for batch in loader:
        # px: (B, 3, 384, 384) images on the device. non_blocking helps the copy overlap with other work (with pinned memory).
        px = batch["pixel_values"].to(device, non_blocking=True)
        # Mixed precision: fp16 math on the GPU is faster and uses less memory (enabled only when amp is True).
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if with_loss:
                # Same forward pass as in training (teacher forcing) to get the validation loss.
                loss = model(pixel_values=px, labels=batch["labels"].to(device)).loss
                # Skip a batch whose loss is NaN / inf so that one bad batch does not ruin the average.
                if torch.isfinite(loss):
                    tot_loss += loss.item()
                    n_batches += 1
            # Real prediction: the model writes the word itself, one character at a time, with no true labels. -> (B, L) ids
            ids = generate_ids(model, px, num_beams=num_beams, max_length=cfg.max_length)
        # Ids -> text. decode() cuts at <eos> and skips special tokens.
        hyps += [tok.decode(r) for r in ids]
        refs += batch["texts"]
        paths += batch["paths"]
    # max(n_batches, 1) avoids division by zero when no batch had a finite loss (or with_loss is False).
    return {"loss": tot_loss / max(n_batches, 1), "cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "refs": refs, "hyps": hyps, "paths": paths}


def run(cfg, resume=False):
    """Full training run. Inputs: cfg (config namespace), resume (continue from last.pt if it exists).
    Returns: dict with best_cer, last_epoch and train_time_min."""
    # Q: What does it do and why does it exist?
    # A: It is the whole training procedure: build data, model, optimizer and scheduler; loop over epochs with
    #    forward, backward and update; validate, log and save checkpoints; stop on time, patience or overfit success.
    # Q: Why does last.pt hold optimizer, scheduler and scaler, and why must total_steps match on resume?
    # A: To continue exactly where it stopped, Adam's moments, the lr position and the fp16 scale are needed, not only
    #    the weights. The lr schedule was built for a fixed number of steps; a different total would give a wrong lr.
    # Fix random seeds so that the run can be repeated.
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mixed precision (fp16) only on a GPU.
    amp = device.type == "cuda"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    tok, loaders = get_loaders(cfg)
    model = build_model(cfg, tok).to(device)
    opt = make_optimizer(model, cfg)
    # One optimizer step happens every grad_accum batches, so there are fewer steps than batches.
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg.grad_accum)
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    # Warm-up is at most 10% of all steps, so that a short run still has time to decay the lr.
    sched = make_scheduler(opt, min(cfg.warmup_steps, max(1, total_steps // 10)), total_steps)
    # GradScaler multiplies the loss by a large number before backward so tiny fp16 gradients do not
    # become zero (underflow). It does nothing when amp is off.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # State that resume has to restore: next epoch, best val CER so far, epochs without improvement, minutes already used.
    start_epoch, best_cer, bad, minutes_before = 0, float("inf"), 0, 0.0
    last_path = os.path.join(cfg.ckpt_dir, "last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        # The lr schedule was built for a fixed number of steps, so the setup must be identical.
        if ck["total_steps"] != total_steps:
            raise ValueError("resume needs the same epochs, batch size and data as the original run: the "
                             "learning-rate schedule was built for a different number of steps")
        # Restore the weights, the optimizer (Adam moments), the scheduler position and the fp16 scaler.
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_cer, bad, minutes_before = ck["epoch"] + 1, ck["best_cer"], ck["bad_epochs"], ck["train_time_min"]
        print(f"resumed from epoch {ck['epoch']} (best val CER so far {best_cer:.4f})", flush=True)

    print(f"device={device} approach={cfg.approach} train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} classes={len(tok)} params={count_params(model):,}", flush=True)
    log_path = os.path.join(cfg.ckpt_dir, "log.csv")
    # A fresh run writes a new log with a header; a resumed run keeps the old log and appends to it.
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_cer", "val_wer", "lr", "minutes"])

    t_start = time.time()
    # Total training minutes, including the minutes used before a resume.
    minutes = lambda: minutes_before + (time.time() - t_start) / 60  # noqa: E731
    time_up, last_epoch = False, start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs):
        # Optionally train only the decoder in the first epochs (the encoder is frozen).
        frozen = epoch < cfg.freeze_encoder_epochs
        set_encoder_trainable(model, not frozen)
        # train() switches dropout on.
        model.train()
        if frozen:
            model.encoder.eval()  # keep the frozen encoder's dropout off
        # run_loss / n_ok: sum and count of finite batch losses, for the epoch's mean training loss.
        run_loss, n_ok = 0.0, 0
        # Clear old gradients. set_to_none=True frees the memory instead of filling it with zeros.
        opt.zero_grad(set_to_none=True)
        n_batches = len(loaders["train"])
        for i, batch in enumerate(loaders["train"]):
            # px (B, 3, 384, 384); labels (B, L) with -100 padding.
            px, labels = batch["pixel_values"].to(device, non_blocking=True), batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                # Forward pass with teacher forcing; the model returns the cross-entropy loss (scalar), and -100 labels are ignored.
                loss = model(pixel_values=px, labels=labels).loss
            # A NaN / inf loss would poison the weights, so such a batch is skipped.
            if torch.isfinite(loss):
                # Gradient accumulation: dividing by grad_accum makes the summed gradients of
                # grad_accum mini-batches equal to their average (a bigger "virtual" batch).
                scaler.scale(loss / cfg.grad_accum).backward()
                run_loss, n_ok = run_loss + loss.item(), n_ok + 1
            # Update the weights every grad_accum batches, and also at the last batch of the epoch so no gradient is left over.
            if (i + 1) % cfg.grad_accum == 0 or i == n_batches - 1:
                # Undo the fp16 loss scaling first, so the clipping below sees the true gradient size.
                scaler.unscale_(opt)
                # Gradient clipping: scale gradients down if their total norm is above grad_clip (guards against exploding gradients).
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                # Optimizer step (skipped by the scaler if gradients contain inf / NaN), then update the scale factor.
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                # The lr schedule moves one step per optimizer update (not per batch).
                sched.step()
            # Time budget: stop this epoch early; the validation and the checkpoint below still run.
            if cfg.max_minutes and minutes() >= cfg.max_minutes:
                time_up = True
                break
        train_loss = run_loss / max(n_ok, 1)

        # Score the model on the validation set after every epoch (greedy decoding).
        res = evaluate_loader(model, loaders["val"], tok, cfg, device, amp)
        # "Best" means lowest validation CER. `bad` counts epochs in a row without improvement (for early stopping).
        improved = res["cer"] < best_cer
        best_cer, bad = (res["cer"], 0) if improved else (best_cer, bad + 1)
        # Last param group = new layers (lr_new), or the only group if there is just one.
        lr_now = opt.param_groups[-1]["lr"]
        print(f"epoch {epoch:3d} | train loss {train_loss:7.3f} | val loss {res['loss']:7.3f} | "
              f"val CER {res['cer']:.4f} | val WER {res['wer']:.4f} | lr {lr_now:.2e} | {minutes():.1f} min"
              + (" | encoder frozen" if frozen else ""), flush=True)
        # Show 3 examples so that a person can see what the model writes.
        for r, h in list(zip(res["refs"], res["hyps"]))[:3]:
            print(f"      truth: {r}   predicted: {h}", flush=True)
        # Append one row to log.csv (used later to plot the training curves).
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{res['loss']:.4f}", f"{res['cer']:.4f}",
                                    f"{res['wer']:.4f}", f"{lr_now:.3e}", f"{minutes():.2f}"])

        # What a checkpoint holds: weights, tokenizer and config (so evaluate.py can rebuild the model alone), and some info.
        state = {"model": model.state_dict(), "tokenizer": tok.to_dict(), "config": to_dict(cfg), "epoch": epoch,
                 "val_cer": res["cer"], "train_time_min": minutes(),
                 "params": count_params(model, trainable_only=False)}
        # best.pt = the weights with the lowest validation CER so far (used for the final test).
        if improved:
            torch.save(state, os.path.join(cfg.ckpt_dir, "best.pt"))
        # last.pt = latest epoch plus optimizer / scheduler / scaler state, so that --resume can continue exactly.
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_cer": best_cer, "bad_epochs": bad,
                    "total_steps": total_steps}, last_path)
        last_epoch = epoch

        # Stop conditions: overfit test passed, time budget used up, or early stopping (patience).
        if cfg.overfit_n > 0 and res["cer"] == 0.0:
            print("overfit test passed: CER is 0 on the training samples", flush=True)
            break
        if time_up:
            print(f"time budget of {cfg.max_minutes} min used up: stopping", flush=True)
            break
        if bad >= cfg.patience:
            print(f"no improvement for {cfg.patience} epochs: early stop", flush=True)
            break

    # best.pt should carry the TOTAL training time, not the time at the best epoch
    best_path = os.path.join(cfg.ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        ck["train_time_min"] = minutes()
        torch.save(ck, best_path)
    return {"best_cer": best_cer, "last_epoch": last_epoch, "train_time_min": minutes()}


def main():
    """Command-line entry point: read the options, build the config and call run().
    Each option overrides the same key from the YAML file; `--set KEY=VALUE` can change any other key."""
    # Q: What does it do and why does it exist?
    # A: It is the command-line entry point. It reads the options, merges them with the YAML config through
    #    load_config, calls run and prints the returned summary dict.
    # Q: Why keep only the options whose value is not None?
    # A: An option the user did not give is None. Passing it on would overwrite the YAML value with None.
    #    Only options that were really typed (and the --set pairs) override the file.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max_minutes", type=float)
    ap.add_argument("--overfit_n", type=int)
    ap.add_argument("--train_limit", type=int)
    ap.add_argument("--data_dir")
    ap.add_argument("--image_root")
    ap.add_argument("--ckpt_dir")
    ap.add_argument("--num_workers", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any other config key")
    a = ap.parse_args()
    # Overrides = the --set pairs, plus every option the user actually gave (None = not given).
    # --config, --resume and --set are not config keys, so they are left out.
    over = {**parse_overrides(a.set), **{k: v for k, v in vars(a).items()
                                         if k not in ("config", "resume", "set") and v is not None}}
    print(run(load_config(a.config, over), resume=a.resume))


if __name__ == "__main__":
    main()
