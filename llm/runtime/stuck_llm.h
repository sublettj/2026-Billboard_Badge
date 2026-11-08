/* stuck_llm: int8 Llama-2 style inference for the badge, endless-monologue edition.
 *
 * A port of llama2.c's runq.c reshaped for a microcontroller:
 *   - no mmap, no stdio, no exit(): the caller hands over the model and
 *     tokenizer bytes (flash-mapped or loaded into PSRAM) and keeps them alive
 *   - all errors are return codes
 *   - the token embedding is read straight from the int8 table, not
 *     dequantized into a 2 MB fp32 copy
 *   - large buffers (KV cache, sampler scratch) go through a pluggable
 *     allocator so the ESP32 side can point them at PSRAM
 *   - stuck_next_piece() runs forever: it starts paragraphs on EOS and slides
 *     the context window when it fills, so the caller just keeps asking
 *
 * Model format: llama2.c export.py version 2 (magic "ak42", Q8_0 groups).
 * Tokenizer format: llama2.c tokenizer.bin.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct stuck_llm stuck_llm_t;

enum {
    STUCK_OK = 0,
    STUCK_ERR_MAGIC = -1,     /* not a version-2 llama2.c model */
    STUCK_ERR_VERSION = -2,
    STUCK_ERR_TRUNCATED = -3, /* model or tokenizer bytes shorter than the header promises */
    STUCK_ERR_NOMEM = -4,
    STUCK_ERR_ARG = -5,
    STUCK_ERR_STATE = -6,     /* tokenizer not loaded, context exhausted in feed, ... */
};

typedef void *(*stuck_alloc_fn)(size_t n);
typedef void (*stuck_free_fn)(void *p);

/* Optional. Routes the big buffers (KV cache, sampler scratch) through these
 * instead of malloc/free. Call before stuck_init(). */
void stuck_set_big_allocator(stuck_alloc_fn alloc, stuck_free_fn free_);

/* Parse the model header and lay the weights over model_bytes in place.
 * model_bytes must stay valid and 4-byte aligned for the life of the model. */
int stuck_init(stuck_llm_t **out, const uint8_t *model_bytes, size_t model_len);

/* Parse a tokenizer.bin. The strings are copied, so tok_bytes may be freed after. */
int stuck_load_tokenizer(stuck_llm_t *m, const uint8_t *tok_bytes, size_t tok_len);

/* temperature 0 = greedy. topp in (0,1) enables nucleus sampling. */
void stuck_set_sampler(stuck_llm_t *m, float temperature, float topp, uint64_t seed);

void stuck_free(stuck_llm_t *m);

/* --- low level ----------------------------------------------------------- */

/* Forget the context. */
void stuck_reset(stuck_llm_t *m);

/* Run the model over tokens without sampling (prompt priming). Fails with
 * STUCK_ERR_STATE if they do not fit in the remaining context. */
int stuck_feed(stuck_llm_t *m, const int *tokens, int n);

/* Forward one token at the current position and sample the next.
 * Returns the sampled token, or STUCK_ERR_STATE if the context is full. */
int stuck_step(stuck_llm_t *m, int token);

/* llama2.c BPE encode. Returns the token count, or <0 on error. */
int stuck_encode(stuck_llm_t *m, const char *text, int bos, int eos, int *tokens, int max_tokens);

/* The UTF-8 piece for a token. For raw-byte tokens the string is one byte long. */
const char *stuck_decode(stuck_llm_t *m, int prev_token, int token);

/* --- the endless monologue ---------------------------------------------- */

/* Optional prompt to start from; otherwise it starts from BOS. */
int stuck_monologue_begin(stuck_llm_t *m, const char *prompt);

/* Produce the next piece of text. Never ends: EOS becomes "\n\n" and a new
 * paragraph starts; a full context is slid back keeping the recent tail.
 * Returns NULL only on an internal error. The string is valid until the next call. */
const char *stuck_next_piece(stuck_llm_t *m);

/* --- introspection -------------------------------------------------------- */
int stuck_vocab_size(const stuck_llm_t *m);
int stuck_seq_len(const stuck_llm_t *m);
int stuck_dim(const stuck_llm_t *m);
int stuck_n_layers(const stuck_llm_t *m);
int stuck_pos(const stuck_llm_t *m);
/* Bytes this instance allocated at runtime (not counting the model bytes). */
size_t stuck_runtime_bytes(const stuck_llm_t *m);

#ifdef __cplusplus
}
#endif
