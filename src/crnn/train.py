"""Train the CRNN with CTC loss.

    python -m src.crnn.train --config configs/crnn/full.yaml
    python -m src.crnn.train --config configs/crnn/full.yaml --resume      # after a Kaggle disconnect

Run from the repository root. A checkpoint is written after every epoch (last.pt) and whenever
the validation CER improves (best.pt). `max_minutes` stops training cleanly when the time budget
is used up, so a run always ends with a usable best.pt.
"""
import argparse
import csv
import os
import time

import torch
import yaml
from torch import nn

from src.common.metrics import cer, wer
from src.common.seed import set_seed

from .dataset import CharVocab, WordDataset, collate_fn
from .decode import greedy_decode
from .model import CRNN, count_params

DEFAULTS = dict(
    data_dir="data/processed/words", train_csv="train.csv", val_csv="val.csv", image_root=".",
    epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4, grad_clip=5.0, num_workers=2, seed=42,
    ckpt_dir="checkpoints/crnn_full", run_name="crnn_full", overfit_n=0, max_minutes=0,
    patience=8, val_limit=0, invert=False, hidden=256, dropout=0.3,
)


def load_config(path=None, **overrides):
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def make_datasets(cfg, vocab):
    d = cfg["data_dir"]
    train = WordDataset(os.path.join(d, cfg["train_csv"]), vocab, root=cfg["image_root"], invert=cfg["invert"])
    val = WordDataset(os.path.join(d, cfg["val_csv"]), vocab, root=cfg["image_root"], invert=cfg["invert"])
    if cfg["overfit_n"] > 0:  # sanity check: learn and test on the same few samples
        train.samples = train.samples[: cfg["overfit_n"]]
        val.samples = train.samples
    elif cfg["val_limit"] > 0:
        val.samples = val.samples[: cfg["val_limit"]]
    return train, val


@torch.no_grad()
def evaluate_loader(model, loader, vocab, device, ctc):
    """Run the model over a loader. Returns loss, CER, WER and the decoded texts."""
    model.eval()
    total_loss, n_batches, refs, hyps, paths = 0.0, 0, [], [], []
    for batch in loader:
        images = batch["images"].to(device)
        out_lens = CRNN.output_lengths(batch["widths"])
        log_probs = model(images).float()
        loss = ctc(log_probs, batch["targets"], out_lens, batch["target_lengths"])
        if torch.isfinite(loss):
            total_loss += float(loss)
            n_batches += 1
        hyps += greedy_decode(log_probs.cpu(), out_lens, vocab)
        refs += batch["texts"]
        paths += batch["paths"]
    return {"loss": total_loss / max(n_batches, 1), "cer": cer(refs, hyps), "wer": wer(refs, hyps),
            "refs": refs, "hyps": hyps, "paths": paths}


def run(cfg, resume=False):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    vocab = CharVocab.from_file(os.path.join(cfg["data_dir"], "vocab.txt"))
    train_ds, val_ds = make_datasets(cfg, vocab)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        collate_fn=collate_fn, pin_memory=use_amp)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"],
        collate_fn=collate_fn, pin_memory=use_amp)

    model = CRNN(len(vocab), hidden=cfg["hidden"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = max(cfg["epochs"] * len(train_loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total_steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ctc = nn.CTCLoss(blank=CharVocab.BLANK, zero_infinity=True)

    start_epoch, best_cer, bad_epochs, minutes_before = 0, float("inf"), 0, 0.0
    last_path = os.path.join(cfg["ckpt_dir"], "last.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck["sched"]["total_steps"] != total_steps:
            raise ValueError("resume needs the same epochs and data as the original run: the learning-rate "
                             "schedule was built for a different number of steps")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_cer = ck["epoch"] + 1, ck["best_cer"]
        bad_epochs, minutes_before = ck["bad_epochs"], ck["train_time_min"]
        print(f"resumed from epoch {ck['epoch']} (best val CER so far {best_cer:.4f})", flush=True)

    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  classes={len(vocab)}  "
          f"params={count_params(model):,}", flush=True)
    log_path = os.path.join(cfg["ckpt_dir"], "log.csv")
    if not (resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_cer", "val_wer", "lr", "minutes"])

    t_start = time.time()
    minutes = lambda: minutes_before + (time.time() - t_start) / 60  # noqa: E731
    time_up = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        run_loss, n_ok = 0.0, 0
        for batch in train_loader:
            images = batch["images"].to(device)
            out_lens = CRNN.output_lengths(batch["widths"])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                log_probs = model(images)
            loss = ctc(log_probs.float(), batch["targets"], out_lens, batch["target_lengths"])
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            n_ok += 1
            if cfg["max_minutes"] and minutes() >= cfg["max_minutes"]:
                time_up = True
                break
        train_loss = run_loss / max(n_ok, 1)

        res = evaluate_loader(model, val_loader, vocab, device, ctc)
        improved = res["cer"] < best_cer
        if improved:
            best_cer, bad_epochs = res["cer"], 0
        else:
            bad_epochs += 1
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | train loss {train_loss:7.3f} | val loss {res['loss']:7.3f} | "
              f"val CER {res['cer']:.4f} | val WER {res['wer']:.4f} | lr {lr_now:.2e} | {minutes():.1f} min", flush=True)
        for r, h in list(zip(res["refs"], res["hyps"]))[:3]:
            print(f"      truth: {r}   predicted: {h}", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{res['loss']:.4f}", f"{res['cer']:.4f}",
                                    f"{res['wer']:.4f}", f"{lr_now:.3e}", f"{minutes():.2f}"])

        state = {"model": model.state_dict(), "chars": vocab.chars, "cfg": cfg, "epoch": epoch,
                 "val_cer": res["cer"], "train_time_min": minutes()}
        if improved:
            torch.save(state, os.path.join(cfg["ckpt_dir"], "best.pt"))
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_cer": best_cer, "bad_epochs": bad_epochs}, last_path)
        last_epoch = epoch

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
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
