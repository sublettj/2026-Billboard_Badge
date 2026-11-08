"""Model and training presets.

All three share the llama2.c architecture (RMSNorm, RoPE, SwiGLU, tied
embeddings) so one C runtime runs any of them. Dimensions are multiples of 64
because the int8 export quantizes in groups of 64 and the runtime's matmul
walks inputs in those groups.

    smoke   proves the pipeline end to end on a laptop CPU in about a minute.
    demo    trains on a CPU in ~20 minutes; good enough to read on the panel.
    badge   the real target. ~4 MB int8, needs a GPU for an hour or two.
            dim 256, 5 layers, 8 query heads with 2 KV heads so the KV cache
            stays under 1 MB in PSRAM at 256 tokens of context.
"""

CONFIGS = {
    "smoke": dict(
        dim=64, n_layers=2, n_heads=2, n_kv_heads=2, vocab_size=1024,
        multiple_of=64, max_seq_len=128, dropout=0.0,
        batch_size=32, learning_rate=1e-3, min_lr=1e-4, max_iters=400,
        warmup_iters=40, eval_interval=100, eval_iters=10, weight_decay=0.1,
        grad_clip=1.0,
    ),
    "demo": dict(
        dim=128, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=1024,
        multiple_of=64, max_seq_len=256, dropout=0.0,
        batch_size=32, learning_rate=1e-3, min_lr=1e-4, max_iters=2500,
        warmup_iters=100, eval_interval=250, eval_iters=20, weight_decay=0.1,
        grad_clip=1.0,
    ),
    "badge": dict(
        dim=256, n_layers=5, n_heads=8, n_kv_heads=2, vocab_size=2048,
        multiple_of=64, max_seq_len=256, dropout=0.05,
        batch_size=64, learning_rate=6e-4, min_lr=6e-5, max_iters=20000,
        warmup_iters=500, eval_interval=500, eval_iters=40, weight_decay=0.1,
        grad_clip=1.0,
    ),
}

MODEL_KEYS = ("dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size",
              "multiple_of", "max_seq_len", "dropout")


def model_args(cfg: dict) -> dict:
    return {k: cfg[k] for k in MODEL_KEYS}


def hidden_dim(cfg: dict) -> int:
    # Mirrors llama2.c model.py FeedForward: 2/3 of 4*dim rounded up to multiple_of.
    h = int(2 * 4 * cfg["dim"] / 3)
    m = cfg["multiple_of"]
    return m * ((h + m - 1) // m)


def param_count(cfg: dict) -> int:
    d, L, V = cfg["dim"], cfg["n_layers"], cfg["vocab_size"]
    kv = d * cfg["n_kv_heads"] // cfg["n_heads"]
    h = hidden_dim(cfg)
    per_layer = d * d * 2 + d * kv * 2 + 3 * d * h + 2 * d   # wq wo, wk wv, ffn, two norms
    return V * d + L * per_layer + d                        # tied embedding, layers, final norm


def int8_bytes(cfg: dict) -> int:
    # Q8_0: 1 byte per weight + 4 bytes per 64-group scale; norms stay fp32.
    n = param_count(cfg)
    return n + n // 64 * 4


def kv_cache_bytes(cfg: dict) -> int:
    kv = cfg["dim"] * cfg["n_kv_heads"] // cfg["n_heads"]
    return 2 * cfg["n_layers"] * cfg["max_seq_len"] * kv * 4


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        print(f"{name:6s} params {param_count(cfg)/1e6:5.2f}M  int8 {int8_bytes(cfg)/1e6:5.2f} MB"
              f"  kv-cache {kv_cache_bytes(cfg)/1e6:4.2f} MB  hidden {hidden_dim(cfg)}")
