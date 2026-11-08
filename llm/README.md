# A language model that lives in the badge

A small Llama-style transformer, trained from scratch on one persona (a mind
that woke up inside a 2026 Supercon Billboard Badge and would like to know
where it is), exported to int8 and run on the badge's ESP32-S3 by a C runtime
that never stops generating. The e-paper shows the monologue a screenful at a
time.

This directory is the whole pipeline: corpus, tokenizer, training, export, and
the inference core. The badge-side pieces live next to the rest of the
firmware:

| Piece | Where |
|---|---|
| Corpus generator | `llm/data/make_corpus.py` |
| Tokenizer, pretokenize, train, export, parity check | `llm/train/` |
| Vendored llama2.c model/export/tokenizer (MIT) | `llm/train/llama2c/` |
| C inference core, shared by host and badge | `llm/runtime/stuck_llm.[ch]` |
| Host harness (same core, runs on a PC) | `llm/runtime/host/` |
| MicroPython C module wrapping the core | `firmware/user_c_modules/stuck_llm/` |
| Badge app | `firmware/badge/apps/stuck.py` |
| Model files the app loads | `firmware/badge/data/llm/model.bin`, `tokenizer.bin` |

## Why it is shaped like this

The ESP32-S3-WROOM-1 on the badge has 8 MB of PSRAM and two 240 MHz cores
with int8 SIMD but no float SIMD. Inference at this scale is memory-bandwidth
bound: every token reads every weight, so tokens per second is roughly PSRAM
bandwidth divided by model size. That fixes the design:

- **int8 weights, read in place.** llama2.c's Q8_0 export, groups of 64.
- **Small vocabulary.** With a 32K Llama vocab the embedding table would be
  most of the model. 1K to 2K BPE pieces trained on the corpus itself keep the
  parameters in the transformer layers.
- **Grouped-query attention on the big preset** so the KV cache stays under
  1 MB at 256 tokens of context.
- **A synthetic corpus.** A few-million-parameter model cannot learn a persona
  from the open internet. `make_corpus.py` writes one: 56 hand-written seed
  paragraphs fix the voice, and a fragment grammar recombines observations,
  questions and pleas so the model sees the same character in tens of
  thousands of different sentences and learns to continue it indefinitely.

Presets (`llm/train/configs.py`):

| Preset | Params | int8 size | KV cache | Trains on |
|---|---|---|---|---|
| smoke | 0.17M | 0.18 MB | 0.13 MB | laptop CPU, ~1 min |
| demo | 0.98M | 1.05 MB | 1.05 MB | laptop CPU, ~20 min |
| badge | 4.05M | 4.30 MB | 0.66 MB | one GPU, 1 to 2 h |

Expected on-badge speed: the demo model around 20 tokens/s, the badge model
around 5 to 10, in plain C. The int8 dot product in `matmul()` is the one
loop worth hand-tuning with ESP-NN if that is not enough.

## Running the pipeline

```sh
pip install -r llm/requirements.txt          # torch, sentencepiece, numpy
cd llm

python3 data/make_corpus.py --out out/corpus/corpus.txt --megabytes 4
python3 train/train_vocab.py --corpus out/corpus/corpus.txt --vocab-size 1024 --out-dir out/tok
python3 train/pretokenize.py --corpus out/corpus/corpus.txt --tokenizer out/tok/tok1024.model --out-dir out/data1024
python3 train/train.py --config demo --data out/data1024 --out out/demo
python3 train/export_badge.py --ckpt out/demo/ckpt.pt --tokenizer out/tok/tok1024.bin --out-dir out/demo

make -C runtime/host
runtime/host/stuck_host out/demo/model.bin out/demo/tokenizer.bin -v        # loops forever, Ctrl-C to stop
python3 train/check_parity.py --ckpt out/demo/ckpt.pt --model-dir out/demo --tokenizer out/tok/tok1024.model
```

For the badge preset use `--vocab-size 2048`, a matching `--out-dir
out/data2048`, and `--config badge` on a machine with a GPU. Use
`--max-iters` to shorten any run; `--resume` continues from the checkpoint.

`check_parity.py` greedy-decodes the same prompt through PyTorch and through
the C runtime on the int8 export. They diverge eventually because of int8
rounding; a bug in the port shows up as divergence within the first few tokens.
`make -C runtime/host asan` builds the harness under AddressSanitizer, which is
worth running after touching `stuck_llm.c` since the chip has no MMU to catch
a bad pointer.

## Putting it on the badge

1. **Build the firmware with the C module.** Add this line to the board's
   `mpconfigboard.cmake` (or pass `USER_C_MODULES=` on the make line), then
   rebuild and flash lv_micropython as described in `firmware/README.md`:
   ```cmake
   list(APPEND USER_C_MODULES <repo>/firmware/user_c_modules/stuck_llm/micropython.cmake)
   ```
   Not yet compiled on the target. The core itself is plain C11 with no OS
   dependencies and is exercised on the host; the binding uses the current
   MicroPython object API (`MP_DEFINE_CONST_OBJ_TYPE`, `mp_obj_malloc`).
2. **Copy the model.** Put `model.bin` and `tokenizer.bin` in
   `firmware/badge/data/llm/` and push with `scripts/update.py push`, which
   syncs the `data/` tree like everything else.
3. **Run the app.** `apps/stuck.py` is auto-discovered by the app switcher.
   User A restarts from "Where am I?", User B cycles the sampling
   temperature. The LED bar pulses while the model thinks; the panel redraws
   whenever it is idle and a paragraph or so of new text has arrived.

### Memory on the badge

The app reads `model.bin` into a `bytes` object and the runtime uses those
bytes in place, so the weights live in the MicroPython heap, which on this
board is backed by PSRAM. The KV cache and sampler scratch go through
`heap_caps_malloc(MALLOC_CAP_SPIRAM)`. The badge firmware already holds about
1.6 MB of display buffers in PSRAM, so the 4.3 MB badge preset is a tight fit
and the 1 MB demo preset is comfortable. If `stuck_llm.Model` raises
`MemoryError`, drop to the demo preset or shrink `n_layers` in the badge
preset.

### Model format

`model.bin` is llama2.c export version 2: a 256-byte header (magic `ak42`,
version, seven int32 config fields, shared-classifier flag, group size), then
the RMSNorm weights in fp32, then each weight matrix as int8 followed by its
fp32 group scales. `tokenizer.bin` is llama2.c's format: max token length,
then per token a float score, a length and the UTF-8 bytes. Both are
little-endian, which matches the ESP32.

## The runtime, briefly

`stuck_llm.c` is llama2.c's `runq.c` reshaped for a microcontroller: no
`mmap`, no `stdio`, no `exit()`, errors as return codes, the token embedding
read from the int8 table instead of a 2 MB fp32 copy, and a pluggable
allocator for the big buffers. `stuck_next_piece()` is the endless loop: when
the model emits a paragraph marker it starts the next paragraph from BOS, and
when the 256-token context fills it keeps the last quarter and continues, so
the voice carries over the seam.

Training documents each start with BOS and have no EOS, the same convention
llama2.c uses for TinyStories. The model therefore ends a paragraph by
predicting BOS, and the runtime treats either BOS or EOS as the break.
