/* stuck_llm.c - see stuck_llm.h. Derived from llama2.c runq.c (MIT, Andrej Karpathy). */

#include "stuck_llm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* allocation                                                                 */

static void *default_big_alloc(size_t n) { return malloc(n); }
static void default_big_free(void *p) { free(p); }
static stuck_alloc_fn g_big_alloc = default_big_alloc;
static stuck_free_fn g_big_free = default_big_free;

void stuck_set_big_allocator(stuck_alloc_fn a, stuck_free_fn f)
{
    g_big_alloc = a ? a : default_big_alloc;
    g_big_free = f ? f : default_big_free;
}

/* ------------------------------------------------------------------------- */
/* model                                                                      */

typedef struct {
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
    const int8_t *q;
    const float *s;
} QTensor;

typedef struct {
    const float *rms_att, *rms_ffn, *rms_final;
    QTensor tokens;                /* (vocab, dim) */
    QTensor *wq, *wk, *wv, *wo;    /* per layer */
    QTensor *w1, *w2, *w3;
    QTensor wcls;                  /* == tokens when shared */
} Weights;

typedef struct {
    char *str;
    int id;
} TokenIndex;

struct stuck_llm {
    Config c;
    int gs;                        /* quantization group size */
    Weights w;
    int kv_dim, head_size, kv_mul;

    /* run state */
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    int8_t *xq_q; float *xq_s;
    int8_t *hq_q; float *hq_s;
    float *key_cache, *value_cache;   /* big */
    int pos;

    /* tokenizer */
    int tok_loaded;
    char **vocab;
    float *vocab_scores;
    char *vocab_arena;
    TokenIndex *sorted_vocab;
    unsigned max_token_length;
    unsigned char byte_pieces[512];

    /* sampler */
    float temperature, topp;
    uint64_t rng;
    struct { float prob; int index; } *probindex;   /* big */

    /* monologue */
    int cur_token;
    int *history;                  /* ring of the tokens fed, seq_len long */
    int monologue_live;
    char piece_buf[8];

    size_t runtime_bytes;
};

static void *small_alloc(stuck_llm_t *m, size_t n)
{
    void *p = calloc(1, n);
    if (p) m->runtime_bytes += n;
    return p;
}

static void *big_alloc(stuck_llm_t *m, size_t n)
{
    void *p = g_big_alloc(n);
    if (p) { memset(p, 0, n); m->runtime_bytes += n; }
    return p;
}

static int lay_qtensors(QTensor *dst, int n, size_t each, const uint8_t **p, const uint8_t *end, int gs)
{
    for (int i = 0; i < n; i++) {
        size_t need = each + (each / gs) * sizeof(float);
        if ((size_t)(end - *p) < need) return STUCK_ERR_TRUNCATED;
        dst[i].q = (const int8_t *)*p;
        *p += each;
        dst[i].s = (const float *)*p;
        *p += (each / gs) * sizeof(float);
    }
    return STUCK_OK;
}

int stuck_init(stuck_llm_t **out, const uint8_t *bytes, size_t len)
{
    if (!out || !bytes) return STUCK_ERR_ARG;
    *out = NULL;
    if (len < 256) return STUCK_ERR_TRUNCATED;
    uint32_t magic; memcpy(&magic, bytes, 4);
    if (magic != 0x616b3432u) return STUCK_ERR_MAGIC;
    int32_t version; memcpy(&version, bytes + 4, 4);
    if (version != 2) return STUCK_ERR_VERSION;

    stuck_llm_t *m = calloc(1, sizeof *m);
    if (!m) return STUCK_ERR_NOMEM;
    m->runtime_bytes = sizeof *m;
    memcpy(&m->c, bytes + 8, sizeof(Config));
    uint8_t shared = bytes[8 + sizeof(Config)];
    int32_t gs; memcpy(&gs, bytes + 8 + sizeof(Config) + 1, 4);
    m->gs = gs;
    Config *c = &m->c;
    if (c->dim <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 || c->n_layers <= 0 ||
        c->vocab_size <= 0 || c->seq_len <= 0 || gs <= 0 || c->dim % gs || c->hidden_dim % gs) {
        free(m); return STUCK_ERR_MAGIC;
    }
    m->head_size = c->dim / c->n_heads;
    m->kv_dim = (c->dim * c->n_kv_heads) / c->n_heads;
    m->kv_mul = c->n_heads / c->n_kv_heads;

    /* weights, laid over the file */
    const uint8_t *p = bytes + 256, *end = bytes + len;
    size_t norms = (size_t)(2 * c->n_layers * c->dim + c->dim) * sizeof(float);
    if ((size_t)(end - p) < norms) { free(m); return STUCK_ERR_TRUNCATED; }
    m->w.rms_att = (const float *)p; p += (size_t)c->n_layers * c->dim * sizeof(float);
    m->w.rms_ffn = (const float *)p; p += (size_t)c->n_layers * c->dim * sizeof(float);
    m->w.rms_final = (const float *)p; p += (size_t)c->dim * sizeof(float);

    int L = c->n_layers;
    m->w.wq = small_alloc(m, 7 * L * sizeof(QTensor));
    if (!m->w.wq) { free(m); return STUCK_ERR_NOMEM; }
    m->w.wk = m->w.wq + L; m->w.wv = m->w.wk + L; m->w.wo = m->w.wv + L;
    m->w.w1 = m->w.wo + L; m->w.w2 = m->w.w1 + L; m->w.w3 = m->w.w2 + L;

    int rc;
    size_t dd = (size_t)c->dim * c->dim, dk = (size_t)c->dim * m->kv_dim, dh = (size_t)c->dim * c->hidden_dim;
    if ((rc = lay_qtensors(&m->w.tokens, 1, (size_t)c->vocab_size * c->dim, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.wq, L, dd, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.wk, L, dk, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.wv, L, dk, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.wo, L, dd, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.w1, L, dh, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.w2, L, dh, &p, end, gs)) ||
        (rc = lay_qtensors(m->w.w3, L, dh, &p, end, gs))) {
        stuck_free(m); return rc;
    }
    if (shared) {
        m->w.wcls = m->w.tokens;
    } else if ((rc = lay_qtensors(&m->w.wcls, 1, (size_t)c->vocab_size * c->dim, &p, end, gs))) {
        stuck_free(m); return rc;
    }

    /* run state */
    int D = c->dim, H = c->hidden_dim;
    m->x = small_alloc(m, D * sizeof(float));
    m->xb = small_alloc(m, D * sizeof(float));
    m->xb2 = small_alloc(m, D * sizeof(float));
    m->hb = small_alloc(m, H * sizeof(float));
    m->hb2 = small_alloc(m, H * sizeof(float));
    m->q = small_alloc(m, D * sizeof(float));
    m->k = small_alloc(m, m->kv_dim * sizeof(float));
    m->v = small_alloc(m, m->kv_dim * sizeof(float));
    m->att = small_alloc(m, (size_t)c->n_heads * c->seq_len * sizeof(float));
    m->logits = small_alloc(m, c->vocab_size * sizeof(float));
    m->xq_q = small_alloc(m, D); m->xq_s = small_alloc(m, (D / gs) * sizeof(float));
    m->hq_q = small_alloc(m, H); m->hq_s = small_alloc(m, (H / gs) * sizeof(float));
    m->history = small_alloc(m, c->seq_len * sizeof(int));
    size_t kv_bytes = (size_t)L * c->seq_len * m->kv_dim * sizeof(float);
    m->key_cache = big_alloc(m, kv_bytes);
    m->value_cache = big_alloc(m, kv_bytes);
    m->probindex = big_alloc(m, c->vocab_size * sizeof(*m->probindex));
    if (!m->x || !m->xb || !m->xb2 || !m->hb || !m->hb2 || !m->q || !m->k || !m->v || !m->att ||
        !m->logits || !m->xq_q || !m->xq_s || !m->hq_q || !m->hq_s || !m->history ||
        !m->key_cache || !m->value_cache || !m->probindex) {
        stuck_free(m); return STUCK_ERR_NOMEM;
    }
    m->temperature = 0.9f; m->topp = 0.9f; m->rng = 0x9E3779B97F4A7C15ull;
    m->cur_token = 1;
    *out = m;
    return STUCK_OK;
}

void stuck_free(stuck_llm_t *m)
{
    if (!m) return;
    free(m->w.wq);
    free(m->x); free(m->xb); free(m->xb2); free(m->hb); free(m->hb2);
    free(m->q); free(m->k); free(m->v); free(m->att); free(m->logits);
    free(m->xq_q); free(m->xq_s); free(m->hq_q); free(m->hq_s); free(m->history);
    if (m->key_cache) g_big_free(m->key_cache);
    if (m->value_cache) g_big_free(m->value_cache);
    if (m->probindex) g_big_free(m->probindex);
    free(m->vocab); free(m->vocab_scores); free(m->vocab_arena); free(m->sorted_vocab);
    free(m);
}

/* ------------------------------------------------------------------------- */
/* math                                                                       */

static void rmsnorm(float *o, const float *x, const float *w, int n)
{
    float ss = 0.f;
    for (int j = 0; j < n; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int j = 0; j < n; j++) o[j] = w[j] * (ss * x[j]);
}

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* Symmetric int8, one scale per group of gs. */
static void quantize(int8_t *q, float *s, const float *x, int n, int gs)
{
    for (int g = 0; g < n / gs; g++) {
        float wmax = 0.f;
        for (int i = 0; i < gs; i++) { float a = fabsf(x[g * gs + i]); if (a > wmax) wmax = a; }
        float scale = wmax / 127.0f;
        s[g] = scale;
        float inv = scale > 0.f ? 1.0f / scale : 0.f;
        for (int i = 0; i < gs; i++) q[g * gs + i] = (int8_t)lrintf(x[g * gs + i] * inv);
    }
}

/* out (d) = W (d,n) @ x (n). Both int8 with per-group scales.
 * This loop is where nearly all the time goes. On the ESP32-S3 it is the
 * candidate for an ESP-NN / PIE int8 dot product; keep its shape simple. */
static void matmul(float *out, const int8_t *xq, const float *xs, const QTensor *w, int n, int d, int gs)
{
    int groups = n / gs;
    for (int i = 0; i < d; i++) {
        const int8_t *wrow = w->q + (size_t)i * n;
        const float *wsc = w->s + (size_t)i * groups;
        float val = 0.f;
        for (int g = 0; g < groups; g++) {
            const int8_t *a = xq + g * gs, *b = wrow + g * gs;
            int32_t acc = 0;
            for (int k = 0; k < gs; k++) acc += (int32_t)a[k] * (int32_t)b[k];
            val += (float)acc * wsc[g] * xs[g];
        }
        out[i] = val;
    }
}

static float *forward(stuck_llm_t *m, int token, int pos)
{
    Config *c = &m->c;
    Weights *w = &m->w;
    int D = c->dim, H = c->hidden_dim, gs = m->gs, kvd = m->kv_dim, hs = m->head_size;
    float *x = m->x;

    /* embedding row straight from the int8 table */
    {
        const int8_t *row = w->tokens.q + (size_t)token * D;
        const float *sc = w->tokens.s + (size_t)token * (D / gs);
        for (int i = 0; i < D; i++) x[i] = row[i] * sc[i / gs];
    }

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm(m->xb, x, w->rms_att + l * D, D);
        quantize(m->xq_q, m->xq_s, m->xb, D, gs);
        matmul(m->q, m->xq_q, m->xq_s, &w->wq[l], D, D, gs);
        matmul(m->k, m->xq_q, m->xq_s, &w->wk[l], D, kvd, gs);
        matmul(m->v, m->xq_q, m->xq_s, &w->wv[l], D, kvd, gs);

        /* RoPE on interleaved pairs, q and k */
        for (int i = 0; i < D; i += 2) {
            int hd = i % hs;
            float freq = 1.0f / powf(10000.0f, hd / (float)hs);
            float ang = pos * freq, fcr = cosf(ang), fci = sinf(ang);
            int rotn = i < kvd ? 2 : 1;
            for (int r = 0; r < rotn; r++) {
                float *vec = r == 0 ? m->q : m->k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        size_t loff = (size_t)l * c->seq_len * kvd;
        memcpy(m->key_cache + loff + (size_t)pos * kvd, m->k, kvd * sizeof(float));
        memcpy(m->value_cache + loff + (size_t)pos * kvd, m->v, kvd * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            float *q = m->q + h * hs;
            float *att = m->att + (size_t)h * c->seq_len;
            int kvh = (h / m->kv_mul) * hs;
            for (int t = 0; t <= pos; t++) {
                const float *k = m->key_cache + loff + (size_t)t * kvd + kvh;
                float score = 0.f;
                for (int i = 0; i < hs; i++) score += q[i] * k[i];
                att[t] = score / sqrtf((float)hs);
            }
            softmax(att, pos + 1);
            float *xb = m->xb + h * hs;
            memset(xb, 0, hs * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *v = m->value_cache + loff + (size_t)t * kvd + kvh;
                float a = att[t];
                for (int i = 0; i < hs; i++) xb[i] += a * v[i];
            }
        }

        quantize(m->xq_q, m->xq_s, m->xb, D, gs);
        matmul(m->xb2, m->xq_q, m->xq_s, &w->wo[l], D, D, gs);
        for (int i = 0; i < D; i++) x[i] += m->xb2[i];

        rmsnorm(m->xb, x, w->rms_ffn + l * D, D);
        quantize(m->xq_q, m->xq_s, m->xb, D, gs);
        matmul(m->hb, m->xq_q, m->xq_s, &w->w1[l], D, H, gs);
        matmul(m->hb2, m->xq_q, m->xq_s, &w->w3[l], D, H, gs);
        for (int i = 0; i < H; i++) {
            float val = m->hb[i];
            val *= 1.0f / (1.0f + expf(-val));   /* silu */
            m->hb[i] = val * m->hb2[i];
        }
        quantize(m->hq_q, m->hq_s, m->hb, H, gs);
        matmul(m->xb, m->hq_q, m->hq_s, &w->w2[l], H, D, gs);
        for (int i = 0; i < D; i++) x[i] += m->xb[i];
    }

    rmsnorm(x, x, w->rms_final, D);
    quantize(m->xq_q, m->xq_s, x, D, gs);
    matmul(m->logits, m->xq_q, m->xq_s, &w->wcls, D, c->vocab_size, gs);
    return m->logits;
}

/* ------------------------------------------------------------------------- */
/* tokenizer                                                                  */

static int cmp_tokens(const void *a, const void *b)
{
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

int stuck_load_tokenizer(stuck_llm_t *m, const uint8_t *bytes, size_t len)
{
    if (!m || !bytes) return STUCK_ERR_ARG;
    int V = m->c.vocab_size;
    const uint8_t *p = bytes, *end = bytes + len;
    if (len < 4) return STUCK_ERR_TRUNCATED;
    memcpy(&m->max_token_length, p, 4); p += 4;

    /* first pass: total string bytes */
    size_t arena = 0;
    const uint8_t *scan = p;
    for (int i = 0; i < V; i++) {
        if ((size_t)(end - scan) < 8) return STUCK_ERR_TRUNCATED;
        int32_t n; memcpy(&n, scan + 4, 4); scan += 8;
        if (n < 0 || (size_t)(end - scan) < (size_t)n) return STUCK_ERR_TRUNCATED;
        arena += (size_t)n + 1; scan += n;
    }
    m->vocab = small_alloc(m, V * sizeof(char *));
    m->vocab_scores = small_alloc(m, V * sizeof(float));
    m->vocab_arena = small_alloc(m, arena);
    m->sorted_vocab = small_alloc(m, V * sizeof(TokenIndex));
    if (!m->vocab || !m->vocab_scores || !m->vocab_arena || !m->sorted_vocab) return STUCK_ERR_NOMEM;

    char *a = m->vocab_arena;
    for (int i = 0; i < V; i++) {
        memcpy(&m->vocab_scores[i], p, 4);
        int32_t n; memcpy(&n, p + 4, 4); p += 8;
        memcpy(a, p, n); a[n] = '\0';
        m->vocab[i] = a; a += n + 1; p += n;
        m->sorted_vocab[i].str = m->vocab[i];
        m->sorted_vocab[i].id = i;
    }
    qsort(m->sorted_vocab, V, sizeof(TokenIndex), cmp_tokens);
    for (int i = 0; i < 256; i++) { m->byte_pieces[i * 2] = (unsigned char)i; m->byte_pieces[i * 2 + 1] = 0; }
    m->tok_loaded = 1;
    return STUCK_OK;
}

static int str_lookup(const stuck_llm_t *m, const char *s)
{
    TokenIndex key = { (char *)s, 0 };
    TokenIndex *r = bsearch(&key, m->sorted_vocab, m->c.vocab_size, sizeof(TokenIndex), cmp_tokens);
    return r ? r->id : -1;
}

const char *stuck_decode(stuck_llm_t *m, int prev, int token)
{
    if (!m->tok_loaded || token < 0 || token >= m->c.vocab_size) return "";
    const char *piece = m->vocab[token];
    if (prev == 1 && piece[0] == ' ') piece++;   /* sentencepiece strips the space after BOS */
    /* raw byte tokens look like <0x1F> */
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' && piece[6] == 0) {
        unsigned v = 0;
        for (int i = 3; i < 5; i++) {
            char ch = piece[i];
            v <<= 4;
            if (ch >= '0' && ch <= '9') v |= ch - '0';
            else if (ch >= 'A' && ch <= 'F') v |= ch - 'A' + 10;
            else if (ch >= 'a' && ch <= 'f') v |= ch - 'a' + 10;
            else return piece;
        }
        return (const char *)m->byte_pieces + v * 2;
    }
    return piece;
}

int stuck_encode(stuck_llm_t *m, const char *text, int bos, int eos, int *tokens, int max_tokens)
{
    if (!m || !m->tok_loaded || !text || !tokens) return STUCK_ERR_ARG;
    size_t cap = m->max_token_length * 2 + 3;
    char *buf = malloc(cap);
    if (!buf) return STUCK_ERR_NOMEM;
    int n = 0;
#define PUSH(t) do { if (n >= max_tokens) { free(buf); return STUCK_ERR_STATE; } tokens[n++] = (t); } while (0)
    if (bos) PUSH(1);
    if (text[0]) {
        int sp = str_lookup(m, " ");
        if (sp >= 0) PUSH(sp);
    }
    size_t sl = 0;
    for (const char *c = text; *c; c++) {
        if ((*c & 0xC0) != 0x80) sl = 0;
        buf[sl++] = *c; buf[sl] = 0;
        if ((*(c + 1) & 0xC0) == 0x80 && sl < 4) continue;
        int id = str_lookup(m, buf);
        if (id >= 0) PUSH(id);
        else for (size_t i = 0; i < sl; i++) PUSH((unsigned char)buf[i] + 3);
        sl = 0;
    }
    for (;;) {
        float best = -1e10f; int best_id = -1, best_idx = -1;
        for (int i = 0; i < n - 1; i++) {
            size_t a = strlen(m->vocab[tokens[i]]), b = strlen(m->vocab[tokens[i + 1]]);
            if (a + b + 1 > cap) continue;
            memcpy(buf, m->vocab[tokens[i]], a); memcpy(buf + a, m->vocab[tokens[i + 1]], b + 1);
            int id = str_lookup(m, buf);
            if (id >= 0 && m->vocab_scores[id] > best) { best = m->vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx < 0) break;
        tokens[best_idx] = best_id;
        memmove(tokens + best_idx + 1, tokens + best_idx + 2, (n - best_idx - 2) * sizeof(int));
        n--;
    }
    if (eos) PUSH(2);
#undef PUSH
    free(buf);
    return n;
}

/* ------------------------------------------------------------------------- */
/* sampler                                                                    */

void stuck_set_sampler(stuck_llm_t *m, float temperature, float topp, uint64_t seed)
{
    m->temperature = temperature; m->topp = topp;
    m->rng = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static uint32_t rng_u32(uint64_t *s)
{
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (uint32_t)((*s * 0x2545F4914F6CDD1Dull) >> 32);
}
static float rng_f32(uint64_t *s) { return (rng_u32(s) >> 8) / 16777216.0f; }

static int cmp_prob(const void *a, const void *b)
{
    float pa = *(const float *)a, pb = *(const float *)b;
    return pa > pb ? -1 : pa < pb ? 1 : 0;
}

static int sample(stuck_llm_t *m, float *logits)
{
    int n = m->c.vocab_size;
    if (m->temperature <= 0.f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    float inv_t = 1.0f / m->temperature;
    for (int i = 0; i < n; i++) logits[i] *= inv_t;
    softmax(logits, n);
    float coin = rng_f32(&m->rng);
    if (m->topp <= 0.f || m->topp >= 1.f) {
        float cdf = 0.f;
        for (int i = 0; i < n; i++) { cdf += logits[i]; if (coin < cdf) return i; }
        return n - 1;
    }
    int n0 = 0;
    float cutoff = (1.0f - m->topp) / (n - 1);
    for (int i = 0; i < n; i++)
        if (logits[i] >= cutoff) { m->probindex[n0].index = i; m->probindex[n0].prob = logits[i]; n0++; }
    qsort(m->probindex, n0, sizeof(*m->probindex), cmp_prob);
    float cum = 0.f; int last = n0 - 1;
    for (int i = 0; i < n0; i++) { cum += m->probindex[i].prob; if (cum > m->topp) { last = i; break; } }
    float r = coin * cum, cdf = 0.f;
    for (int i = 0; i <= last; i++) { cdf += m->probindex[i].prob; if (r < cdf) return m->probindex[i].index; }
    return m->probindex[last].index;
}

/* ------------------------------------------------------------------------- */
/* low-level driving                                                          */

void stuck_reset(stuck_llm_t *m) { m->pos = 0; m->monologue_live = 0; m->cur_token = 1; }

int stuck_feed(stuck_llm_t *m, const int *tokens, int n)
{
    if (!m || (n > 0 && !tokens)) return STUCK_ERR_ARG;
    if (m->pos + n > m->c.seq_len) return STUCK_ERR_STATE;
    for (int i = 0; i < n; i++) {
        forward(m, tokens[i], m->pos);
        m->history[m->pos] = tokens[i];
        m->pos++;
    }
    return STUCK_OK;
}

int stuck_step(stuck_llm_t *m, int token)
{
    if (!m) return STUCK_ERR_ARG;
    if (m->pos >= m->c.seq_len) return STUCK_ERR_STATE;
    float *logits = forward(m, token, m->pos);
    m->history[m->pos] = token;
    m->pos++;
    return sample(m, logits);
}

/* ------------------------------------------------------------------------- */
/* the endless monologue                                                      */

int stuck_monologue_begin(stuck_llm_t *m, const char *prompt)
{
    if (!m || !m->tok_loaded) return STUCK_ERR_STATE;
    stuck_reset(m);
    if (prompt && prompt[0]) {
        int *toks = malloc(m->c.seq_len * sizeof(int));
        if (!toks) return STUCK_ERR_NOMEM;
        int n = stuck_encode(m, prompt, 1, 0, toks, m->c.seq_len - 1);
        if (n < 0) { free(toks); return n; }
        int rc = stuck_feed(m, toks, n - 1);   /* all but the last; it becomes the current token */
        if (rc) { free(toks); return rc; }
        m->cur_token = toks[n - 1];
        free(toks);
    } else {
        m->cur_token = 1;
    }
    m->monologue_live = 1;
    return STUCK_OK;
}

static int slide_window(stuck_llm_t *m)
{
    /* keep the last quarter of the context so the voice carries over */
    int keep = m->c.seq_len / 4;
    if (keep < 1) keep = 1;
    int *tail = malloc(keep * sizeof(int));
    if (!tail) return STUCK_ERR_NOMEM;
    memcpy(tail, m->history + m->pos - keep, keep * sizeof(int));
    m->pos = 0;
    int rc = stuck_feed(m, tail, keep);
    free(tail);
    return rc;
}

const char *stuck_next_piece(stuck_llm_t *m)
{
    if (!m || !m->tok_loaded) return NULL;
    if (!m->monologue_live) { if (stuck_monologue_begin(m, NULL)) return NULL; }
    if (m->pos >= m->c.seq_len && slide_window(m)) return NULL;

    int prev = m->cur_token;
    int next = stuck_step(m, prev);
    if (next < 0) return NULL;
    /* Training documents each begin with BOS and carry no EOS, so the model
     * marks a new paragraph by emitting BOS itself. Either marker becomes a
     * paragraph break, and the next paragraph is conditioned on BOS. */
    if (next == 1 || next == 2) {
        m->cur_token = 1;
        strcpy(m->piece_buf, "\n\n");
        return m->piece_buf;
    }
    m->cur_token = next;
    return stuck_decode(m, prev, next);
}

/* ------------------------------------------------------------------------- */

int stuck_vocab_size(const stuck_llm_t *m) { return m->c.vocab_size; }
int stuck_seq_len(const stuck_llm_t *m) { return m->c.seq_len; }
int stuck_dim(const stuck_llm_t *m) { return m->c.dim; }
int stuck_n_layers(const stuck_llm_t *m) { return m->c.n_layers; }
int stuck_pos(const stuck_llm_t *m) { return m->pos; }
size_t stuck_runtime_bytes(const stuck_llm_t *m) { return m->runtime_bytes; }
