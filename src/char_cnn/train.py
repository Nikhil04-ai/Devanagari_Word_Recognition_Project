"""Train the character CNN (Model A) on DHCD.

    python -m src.char_cnn.train --config configs/char_cnn/full.yaml
    python -m src.char_cnn.train --config configs/char_cnn/full.yaml --resume   # after a disconnect

Run from the repository root. 10 % of the official Train folder (stratified, seed 42) is the
validation set; the Test folder is not touched here (src.char_cnn.evaluate uses it once).
A checkpoint is written after every epoch (last.pt) and whenever validation accuracy improves
(best.pt). `max_minutes` stops training cleanly when the time budget is used up.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import yaml
from torch import nn

from src.common.seed import set_seed

from .dataset import CHARS, NUM_CLASSES, load_split, random_affine, stratified_split, to_input
from .model import CharCNN, count_params

DEFAULTS = dict(
    data_dir="data/raw/dhcd", val_fraction=0.1, epochs=30, batch_size=128, lr=1e-3, weight_decay=1e-4,
    label_smoothing=0.1, dropout=0.4, augment=True, seed=42, ckpt_dir="checkpoints/char_cnn_full",
    run_name="char_cnn_full", overfit_n=0, max_minutes=0, patience=8,
)


def load_config(path=None, **overrides):
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def make_data(cfg):
    """Return (x_train, y_train, x_val, y_val) as uint8 images and int64 labels (CPU tensors)."""
    images, labels, _ = load_split(cfg["data_dir"], "Train")
    tr, va = stratified_split(labels.numpy(), cfg["val_fraction"], cfg["seed"])
    if cfg["overfit_n"] > 0:  # sanity check: learn and test on the same few samples
        rng = np.random.default_rng(cfg["seed"])
        tr = np.sort(rng.choice(tr, size=min(cfg["overfit_n"], len(tr)), replace=False))
        va = tr
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)
    return images[tr], labels[tr], images[va], labels[va]


@torch.no_grad()
def evaluate(model, x_uint8, y, device, criterion, batch_size=1024):
    """Loss, accuracy and predicted class indices for a whole set."""
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(y), batch_size):
        xb = to_input(x_uint8[i:i + batch_size].to(device))
        yb = y[i:i + batch_size].to(device)
        logits = model(xb).float()
        total_loss += float(criterion(logits, yb)) * len(yb)
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    acc = float((preds == y).float().mean()) if len(y) else 0.0
    return {"loss": total_loss / max(len(y), 1), "acc": acc, "preds": preds}


def run(cfg, resume=False):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    x_tr, y_tr, x_va, y_va = make_data(cfg)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)  # about 70 MB: fits easily on the GPU

    model = CharCNN(NUM_CLASSES, dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    eval_loss_fn = nn.CrossEntropyLoss()

    start_epoch, best_acc, bad_epochs, minutes_before = 0, -1.0, 0, 0.0
    last_path = os.path.join(cfg["ckpt_dir"], "last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["cfg"]["epochs"] != cfg["epochs"]:
            raise ValueError("resume needs the same 'epochs' as the original run (the cosine schedule depends on it)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        bad_epochs, minutes_before = ck["bad_epochs"], ck["train_time_min"]
        print(f"resumed from epoch {ck['epoch']} (best val accuracy so far {best_acc:.4f})", flush=True)

    print(f"device={device}  train={len(y_tr)}  val={len(y_va)}  classes={NUM_CLASSES}  "
          f"params={count_params(model):,}", flush=True)
    log_path = os.path.join(cfg["ckpt_dir"], "log.csv")
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "minutes"])

    t_start = time.time()
    minutes = lambda: minutes_before + (time.time() - t_start) / 60  # noqa: E731
    augment = cfg["augment"] and cfg["overfit_n"] == 0
    time_up, last_epoch = False, start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(y_tr), device=device)
        run_loss, correct, seen = 0.0, 0, 0
        for i in range(0, len(perm), cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            xb, yb = to_input(x_tr[idx]), y_tr[idx]
            if augment:
                xb = random_affine(xb)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
            loss = train_loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(yb)
            if cfg["max_minutes"] and minutes() >= cfg["max_minutes"]:
                time_up = True
                break
        sched.step()
        train_loss, train_acc = run_loss / max(seen, 1), correct / max(seen, 1)

        res = evaluate(model, x_va, y_va, device, eval_loss_fn)
        improved = res["acc"] > best_acc
        if improved:
            best_acc, bad_epochs = res["acc"], 0
        else:
            bad_epochs += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {res['loss']:.4f} acc {res['acc']:.4f} | lr {lr_now:.2e} | {minutes():.1f} min", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{res['loss']:.4f}",
                                    f"{res['acc']:.4f}", f"{lr_now:.3e}", f"{minutes():.2f}"])

        state = {"model": model.state_dict(), "chars": CHARS, "cfg": cfg, "epoch": epoch,
                 "val_acc": res["acc"], "train_time_min": minutes()}
        if improved:
            torch.save(state, os.path.join(cfg["ckpt_dir"], "best.pt"))
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_acc": best_acc, "bad_epochs": bad_epochs}, last_path)
        last_epoch = epoch

        if cfg["overfit_n"] > 0 and res["acc"] == 1.0:
            print("overfit test passed: accuracy is 1.0 on the training samples", flush=True)
            break
        if time_up:
            print(f"time budget of {cfg['max_minutes']} min used up: stopping", flush=True)
            break
        if bad_epochs >= cfg["patience"]:
            print(f"no improvement for {cfg['patience']} epochs: early stop", flush=True)
            break

    return {"best_val_acc": best_acc, "last_epoch": last_epoch, "train_time_min": minutes()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
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
