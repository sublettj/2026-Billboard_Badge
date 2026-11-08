#!/usr/bin/env python3
"""Tokenize the corpus into flat uint16 token streams for training.

Each blank-line-separated paragraph is one document and gets a BOS token in
front of it, exactly as llama2.c does for TinyStories. The on-device loop
relies on this: when the model emits EOS it starts the next paragraph by
feeding BOS, and because BOS appeared mid-stream throughout training the model
knows how to begin a fresh thought from there.

    python3 pretokenize.py --corpus out/corpus/corpus.txt --tokenizer out/tok/tok1024.model --out-dir out/data1024
"""

import argparse
import os
import random

import numpy as np
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokenizer", required=True, help="sentencepiece .model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    bos = sp.bos_id()
    assert sp.vocab_size() < 65536

    with open(args.corpus, encoding="utf-8") as f:
        text = f.read()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(paras)
    n_val = max(1, int(len(paras) * args.val_fraction))
    splits = {"val": paras[:n_val], "train": paras[n_val:]}

    os.makedirs(args.out_dir, exist_ok=True)
    total_chars = 0
    for name, docs in splits.items():
        ids = []
        for p in docs:
            ids.append(bos)
            ids.extend(sp.encode(p))
            total_chars += len(p)
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(args.out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"{name}: {len(docs)} docs, {len(arr)} tokens -> {path}")
    n_tok = sum(os.path.getsize(os.path.join(args.out_dir, f"{s}.bin")) // 2 for s in splits)
    print(f"{total_chars / n_tok:.2f} chars per token")


if __name__ == "__main__":
    main()
