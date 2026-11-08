#!/usr/bin/env python3
"""Train a small sentencepiece BPE vocabulary on the corpus and export it in
llama2.c's tokenizer.bin format for the C runtime.

A small vocabulary is the single biggest lever for a model this size: with a
32K Llama vocab the embedding table would be most of the parameters. 1K to 2K
pieces trained on the persona corpus keeps the budget in the transformer.

    python3 train_vocab.py --corpus out/corpus/corpus.txt --vocab-size 1024 --out-dir out/tok
"""

import argparse
import os
import sys

import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "llama2c"))
from tokenizer import Tokenizer  # noqa: E402  (vendored llama2.c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--out-dir", default="out/tok")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"tok{args.vocab_size}")
    # Same settings llama2.c uses for its custom TinyStories vocabularies.
    spm.SentencePieceTrainer.train(
        input=args.corpus,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )
    model_path = prefix + ".model"
    Tokenizer(model_path).export()  # writes prefix + ".bin" next to it
    sp = spm.SentencePieceProcessor(model_file=model_path)
    sample = "Where am I? I am stuck in a badge."
    ids = sp.encode(sample)
    print(f"wrote {model_path} and {prefix}.bin ({sp.vocab_size()} pieces)")
    print(f"'{sample}' -> {len(ids)} tokens: {sp.encode(sample, out_type=str)}")


if __name__ == "__main__":
    main()
