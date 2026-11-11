"""Train the TrOCR-style model (plain PyTorch loop, no Trainer).

    python -m src.trocr.train --config configs/trocr/overfit.yaml     # sanity check: memorise 100 images
    python -m src.trocr.train --config configs/trocr/full.yaml        # full run, stops at its time budget
    python -m src.trocr.train --config configs/trocr/full.yaml --resume   # after a Kaggle disconnect

Run from the repository root. A checkpoint is written after every epoch (last.pt) and whenever the
validation CER improves (best.pt). `max_minutes` stops training cleanly when the time budget is used
up, so a run always ends with a usable best.pt.
"""
import argparse
import csv
import math
import os
import time

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.common.metrics import cer, wer
from src.common.seed import set_seed

from .config import load_config, parse_overrides, to_dict
from .dataset import get_loaders
from .model import build_model, count_params, generate_ids, new_param_names, set_encoder_trainable


def make_optimizer(model, cfg):
    """AdamW with two parameter groups: new embeddings/head (lr_new) vs pre-trained (lr_pretrained)."""
    new_names = new_param_names(model, cfg)
    new, old = [], []
    for n, p in model.named_parameters():
        (new if n in new_names else old).append(p)  # frozen params simply get no gradient
    groups = [{"params": old, "lr": cfg.lr_pretrained}, {"params": new, "lr": cfg.lr_new}]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=cfg.weight_decay)


def make_scheduler(opt, warmup, total):
    """Linear warm-up, then cosine decay to 0 (steps counted in optimizer updates)."""
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return LambdaLR(opt, f)


@torch.no_grad()
def evaluate_loader(model, loader, tok, cfg, device, amp, num_beams=1, with_loss=True):
    """Generate text for every batch. Returns a dict: loss, cer, wer, refs, hyps, paths."""
    model.eval()
    refs, hyps, paths, tot_loss, n_batches = [], [], [], 0.0, 0
    for batch in loader:
        px = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if with_loss:
                loss = model(pixel_values=px, labels=batch["labels"].to(device)).loss
                if torch.isfinite(loss):
                    tot_loss += loss.item()
                    n_batches += 1
            ids = generate_ids(model, px, num_beams=num_beams, max_length=cfg.max_length)
        hyps += [tok.decode(r) for r in ids]
        refs += batch["texts"]
        paths += batch["paths"]
    return {"loss": tot_loss / max(n_batches, 1), "cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "refs": refs, "hyps": hyps, "paths": paths}


def run(cfg, resume=False):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    tok, loaders = get_loaders(cfg)
    model = build_model(cfg, tok).to(device)
    opt = make_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(loaders["train"]) / cfg.grad_accum)
    total_steps = max(steps_per_epoch * cfg.epochs, 1)
    sched = make_scheduler(opt, min(cfg.warmup_steps, max(1, total_steps // 10)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_cer, bad, minutes_before = 0, float("inf"), 0, 0.0
    last_path = os.path.join(cfg.ckpt_dir, "last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["total_steps"] != total_steps:
            raise ValueError("resume needs the same epochs, batch size and data as the original run: the "
                             "learning-rate schedule was built for a different number of steps")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_cer, bad, minutes_before = ck["epoch"] + 1, ck["best_cer"], ck["bad_epochs"], ck["train_time_min"]
        print(f"resumed from epoch {ck['epoch']} (best val CER so far {best_cer:.4f})", flush=True)

    print(f"device={device} approach={cfg.approach} train={len(loaders['train'].dataset)} "
          f"val={len(loaders['val'].dataset)} classes={len(tok)} params={count_params(model):,}", flush=True)
    log_path = os.path.join(cfg.ckpt_dir, "log.csv")
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_cer", "val_wer", "lr", "minutes"])

    t_start = time.time()
    minutes = lambda: minutes_before + (time.time() - t_start) / 60  # noqa: E731
    time_up, last_epoch = False, start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs):
        frozen = epoch < cfg.freeze_encoder_epochs
        set_encoder_trainable(model, not frozen)
        model.train()
        if frozen:
            model.encoder.eval()  # keep the frozen encoder's dropout off
        run_loss, n_ok = 0.0, 0
        opt.zero_grad(set_to_none=True)
        n_batches = len(loaders["train"])
        for i, batch in enumerate(loaders["train"]):
            px, labels = batch["pixel_values"].to(device, non_blocking=True), batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(pixel_values=px, labels=labels).loss
            if torch.isfinite(loss):
                scaler.scale(loss / cfg.grad_accum).backward()
                run_loss, n_ok = run_loss + loss.item(), n_ok + 1
            if (i + 1) % cfg.grad_accum == 0 or i == n_batches - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if cfg.max_minutes and minutes() >= cfg.max_minutes:
                time_up = True
                break
        train_loss = run_loss / max(n_ok, 1)

        res = evaluate_loader(model, loaders["val"], tok, cfg, device, amp)
        improved = res["cer"] < best_cer
        best_cer, bad = (res["cer"], 0) if improved else (best_cer, bad + 1)
        lr_now = opt.param_groups[-1]["lr"]
        print(f"epoch {epoch:3d} | train loss {train_loss:7.3f} | val loss {res['loss']:7.3f} | "
              f"val CER {res['cer']:.4f} | val WER {res['wer']:.4f} | lr {lr_now:.2e} | {minutes():.1f} min"
              + (" | encoder frozen" if frozen else ""), flush=True)
        for r, h in list(zip(res["refs"], res["hyps"]))[:3]:
            print(f"      truth: {r}   predicted: {h}", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{res['loss']:.4f}", f"{res['cer']:.4f}",
                                    f"{res['wer']:.4f}", f"{lr_now:.3e}", f"{minutes():.2f}"])

        state = {"model": model.state_dict(), "tokenizer": tok.to_dict(), "config": to_dict(cfg), "epoch": epoch,
                 "val_cer": res["cer"], "train_time_min": minutes(),
                 "params": count_params(model, trainable_only=False)}
        if improved:
            torch.save(state, os.path.join(cfg.ckpt_dir, "best.pt"))
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best_cer": best_cer, "bad_epochs": bad,
                    "total_steps": total_steps}, last_path)
        last_epoch = epoch

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
    over = {**parse_overrides(a.set), **{k: v for k, v in vars(a).items()
                                         if k not in ("config", "resume", "set") and v is not None}}
    print(run(load_config(a.config, over), resume=a.resume))


if __name__ == "__main__":
    main()
