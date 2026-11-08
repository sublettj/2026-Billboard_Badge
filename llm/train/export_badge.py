#!/usr/bin/env python3
"""Export a checkpoint to the int8 model.bin the C runtime loads, and put the
matching tokenizer.bin next to it.

    python3 export_badge.py --ckpt out/smoke/ckpt.pt --tokenizer out/tok/tok1024.bin --out-dir out/smoke
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "llama2c"))
from export import load_checkpoint, version2_export  # noqa: E402  (vendored llama2.c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer .bin from train_vocab.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_checkpoint(args.ckpt)
    model_path = os.path.join(args.out_dir, "model.bin")
    version2_export(model, model_path, group_size=args.group_size)
    tok_path = os.path.join(args.out_dir, "tokenizer.bin")
    shutil.copyfile(args.tokenizer, tok_path)
    p = model.params
    print(f"model.bin {os.path.getsize(model_path)/1e6:.2f} MB, tokenizer.bin {os.path.getsize(tok_path)/1e3:.0f} KB")
    print(f"dim {p.dim} layers {p.n_layers} heads {p.n_heads}/{p.n_kv_heads or p.n_heads} vocab {p.vocab_size} seq {p.max_seq_len}")


if __name__ == "__main__":
    main()
