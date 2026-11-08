/* Host harness for stuck_llm: runs the same core the badge runs, on a PC.
 *
 *   ./stuck_host model.bin tokenizer.bin [-n tokens] [-t temp] [-p topp] [-s seed] [-i "prompt"] [-v]
 *
 * -n 0 (the default) loops forever, which is the badge's job description.
 */
#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../stuck_llm.h"

static uint8_t *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)n + 4);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *len = (size_t)n;
    return buf;
}

static void safe_print(const char *piece)
{
    if (!piece || !piece[0]) return;
    if (!piece[1] && !(isprint((unsigned char)piece[0]) || isspace((unsigned char)piece[0]))) return;
    fputs(piece, stdout);
}

static double now(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tokenizer.bin [-n tokens] [-t temp] [-p topp] [-s seed] [-i prompt] [-v]\n", argv[0]); return 2; }
    long max_tokens = 0; float temp = 0.9f, topp = 0.9f; uint64_t seed = (uint64_t)time(NULL);
    const char *prompt = NULL; int verbose = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tokens = atol(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) topp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
    }

    size_t mlen, tlen;
    uint8_t *model = slurp(argv[1], &mlen);
    uint8_t *tok = slurp(argv[2], &tlen);
    if (!model || !tok) return 1;

    stuck_llm_t *m;
    int rc = stuck_init(&m, model, mlen);
    if (rc) { fprintf(stderr, "stuck_init failed: %d\n", rc); return 1; }
    rc = stuck_load_tokenizer(m, tok, tlen);
    if (rc) { fprintf(stderr, "tokenizer failed: %d\n", rc); return 1; }
    free(tok);
    stuck_set_sampler(m, temp, topp, seed);
    if (verbose)
        fprintf(stderr, "model: dim %d layers %d vocab %d seq %d, %.2f MB weights, %zu KB runtime\n",
                stuck_dim(m), stuck_n_layers(m), stuck_vocab_size(m), stuck_seq_len(m),
                mlen / 1e6, stuck_runtime_bytes(m) / 1024);

    rc = stuck_monologue_begin(m, prompt);
    if (rc) { fprintf(stderr, "begin failed: %d\n", rc); return 1; }
    if (prompt) { fputs(prompt, stdout); }

    double t0 = now(); long n = 0;
    for (;;) {
        const char *piece = stuck_next_piece(m);
        if (!piece) { fprintf(stderr, "\ngeneration error\n"); break; }
        safe_print(piece);
        fflush(stdout);
        n++;
        if (verbose && n % 200 == 0) fprintf(stderr, "\n[%ld tokens, %.1f tok/s, pos %d]\n", n, n / (now() - t0), stuck_pos(m));
        if (max_tokens && n >= max_tokens) break;
    }
    if (verbose) fprintf(stderr, "\n%ld tokens in %.2fs = %.1f tok/s\n", n, now() - t0, n / (now() - t0));
    putchar('\n');
    stuck_free(m);
    free(model);
    return 0;
}
