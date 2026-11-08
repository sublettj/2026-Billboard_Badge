#!/usr/bin/env python3
"""Compare greedy generation from the PyTorch checkpoint against the C runtime
on the exported int8 model. They will not match forever (int8 rounding), but a
porting bug in RoPE, attention or the tokenizer shows up as divergence within
the first handful of tokens, so this is the smoke test to run after touching
stuck_llm.c.

    python3 check_parity.py --ckpt out/smoke/ckpt.pt --model-dir out/smoke --tokenizer out/tok/tok1024.model
"""

import argparse
import os
import subprocess
import sys

import sentencepiece as spm
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "llama2c"))
from export import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-dir", required=True, help="dir with model.bin and tokenizer.bin")
    ap.add_argument("--tokenizer", required=True, help="sentencepiece .model")
    ap.add_argument("--prompt", default="Where am I? I am")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--host-bin", default=os.path.join(HERE, "..", "runtime", "host", "stuck_host"))
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = load_checkpoint(args.ckpt)
    ids = [sp.bos_id()] + sp.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.tokens, temperature=0.0)
    torch_text = sp.decode(y[0, len(ids):].tolist())

    out = subprocess.run(
        [args.host_bin, os.path.join(args.model_dir, "model.bin"), os.path.join(args.model_dir, "tokenizer.bin"),
         "-t", "0", "-n", str(args.tokens), "-i", args.prompt],
        capture_output=True, text=True, check=True).stdout
    c_text = out[len(args.prompt):].strip()
    torch_text = torch_text.strip()

    print("torch :", torch_text)
    print("c int8:", c_text)
    agree = 0
    for a, b in zip(torch_text, c_text):
        if a != b:
            break
        agree += 1
    frac = agree / max(1, len(torch_text))
    print(f"identical prefix: {agree} chars ({frac:.0%} of torch output)")
    if frac < 0.5:
        print("FAIL: outputs diverge early; suspect a runtime bug")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
