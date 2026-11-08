#!/usr/bin/env python3
"""Train one of the presets in configs.py on the pretokenized corpus.

Single-process, runs on CPU or one GPU. Writes out/<name>/ckpt.pt in the
checkpoint layout llama2.c's export.py expects, so export_badge.py can turn it
straight into model.bin.

    python3 train.py --config smoke --data out/data1024 --out out/smoke
    python3 train.py --config badge --data out/data2048 --out out/badge --max-iters 20000
"""

import argparse
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "llama2c"))
from model import ModelArgs, Transformer  # noqa: E402  (vendored llama2.c)
from configs import CONFIGS, model_args, param_count, int8_bytes  # noqa: E402


class Batches:
    def __init__(self, data_dir, seq_len, batch_size, device):
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device

    def get(self, split):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + self.seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + self.seq_len].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)


def lr_at(it, cfg):
    if it < cfg["warmup_iters"]:
        return cfg["learning_rate"] * (it + 1) / cfg["warmup_iters"]
    if it >= cfg["max_iters"]:
        return cfg["min_lr"]
    ratio = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * ratio)) * (cfg["learning_rate"] - cfg["min_lr"])


@torch.no_grad()
def estimate_loss(model, batches, eval_iters, ctx):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = batches.get(split)
            with ctx:
                model(x, y)
            losses[k] = model.last_loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS.keys(), required=True)
    ap.add_argument("--data", required=True, help="dir with train.bin and val.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iters", type=int, help="override the preset")
    ap.add_argument("--batch-size", type=int, help="override the preset")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.config])
    if args.max_iters:
        cfg["max_iters"] = args.max_iters
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"
    # bf16 autocast on GPU; plain fp32 on CPU where it is both faster and simpler.
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else torch.autocast("cpu", enabled=False)

    margs = ModelArgs(**model_args(cfg))
    model = Transformer(margs)
    iter_num, best_val = 0, float("inf")
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        iter_num, best_val = ck["iter_num"], ck["best_val_loss"]
        print(f"resumed from iter {iter_num}, best val {best_val:.3f}")
    model.to(device)
    print(f"{args.config}: {param_count(cfg)/1e6:.2f}M params, ~{int8_bytes(cfg)/1e6:.2f} MB int8, device {device}")

    optimizer = model.configure_optimizers(cfg["weight_decay"], cfg["learning_rate"], (0.9, 0.95), device_type)
    if args.resume and os.path.exists(ckpt_path) and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    if args.compile:
        model = torch.compile(model)

    batches = Batches(args.data, cfg["max_seq_len"], cfg["batch_size"], device)
    tokens_per_iter = cfg["batch_size"] * cfg["max_seq_len"]
    x, y = batches.get("train")
    t0 = time.time()
    while iter_num <= cfg["max_iters"]:
        lr = lr_at(iter_num, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if iter_num % cfg["eval_interval"] == 0 or iter_num == cfg["max_iters"]:
            losses = estimate_loss(model, batches, cfg["eval_iters"], ctx)
            print(f"iter {iter_num}: train {losses['train']:.3f}  val {losses['val']:.3f}  lr {lr:.2e}")
            if losses["val"] < best_val or iter_num == cfg["max_iters"]:
                best_val = min(best_val, losses["val"])
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": asdict(margs),
                    "iter_num": iter_num,
                    "best_val_loss": best_val,
                    "config": cfg,
                }, ckpt_path)
        if iter_num == cfg["max_iters"]:
            break

        with ctx:
            model(x, y)
            loss = model.last_loss
        x, y = batches.get("train")  # prefetch while backward runs on GPU
        loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1
        if iter_num % 20 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"  iter {iter_num}  loss {loss.item():.3f}  {20 * tokens_per_iter / dt / 1e3:.0f}k tok/s")

    print(f"done. best val loss {best_val:.3f}, checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
