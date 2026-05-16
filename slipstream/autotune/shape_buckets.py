"""Shape bucketing: discrete grid the autotuner uses.

Without bucketing, every micro-shape would be a cache miss. With bucketing,
shapes that differ trivially (M=33 vs M=32, ctx=2049 vs ctx=2048) map to the
same canonical bucket and reuse the same tuned config.

Decode-step shapes cluster around a small set of "real" values:

  - Batch sizes are usually 1, 4, 8, 16, 32, 64, 128, 256 (powers of 2).
  - Context lengths are powers of 2 with some 1.5× points (512, 1024, 1536,
    2048, ...).
  - Head dim is one of {64, 128} for ~all production models.
  - Hidden / intermediate dims are model constants — not bucketed.

The buckets are intentionally coarser than necessary; sweeping a finer grid
gains < 5% in measured perf and costs autotune time linearly.
"""

from __future__ import annotations

import bisect


def _bucket_pow2_ceil(x: int, _table: list[int] | None = None) -> int:
    """Round up to the nearest entry in a power-of-two-flavored grid."""
    if _table is None:
        _table = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768,
                  1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
                  24576, 32768, 49152, 65536]
    if x <= _table[0]:
        return _table[0]
    if x >= _table[-1]:
        return _table[-1]
    return _table[bisect.bisect_left(_table, x)]


def bucket_attention_shape(
    *,
    batch: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    max_ctx_len: int,
    dtype: str,
    fp8_kv: bool,
) -> str:
    """Canonical bucket string for a paged-attn decode call.

    ``max_ctx_len`` is the maximum sequence length in the batch — we tune for
    the worst-case sequence since that dominates kernel runtime.
    """
    b = _bucket_pow2_ceil(batch)
    ctx = _bucket_pow2_ceil(max_ctx_len)
    # head_dim and head counts are typically exact (model constants),
    # so we don't bucket them.
    fp8_tag = "fp8" if fp8_kv else "fp16"
    return (
        f"b{b}_qh{n_q_heads}_kvh{n_kv_heads}_d{head_dim}_ctx{ctx}_"
        f"{dtype}_{fp8_tag}"
    )


def bucket_gemm_shape(
    *,
    M: int,
    N: int,
    K: int,
    dtype: str,
    quant: str = "fp8",
) -> str:
    """Canonical bucket string for a decode-step GEMM call.

    ``N`` and ``K`` are model constants (hidden / intermediate sizes) so we
    bucket M only. Skinny decode regime is M ∈ {1, 4, 16, 32, ...}.
    """
    m = _bucket_pow2_ceil(M)
    return f"m{m}_n{N}_k{K}_{dtype}_{quant}"


def attention_decode_suite(
    head_dim: int = 128,
    n_q_heads: int = 32,
    n_kv_heads: int = 8,
    dtype: str = "fp16",
) -> list[dict]:
    """The decode-shape grid the autotune CLI sweeps by default.

    Matches a Llama-3-8B-style configuration. Per-shape settings, not bucket
    strings (the buckets are derived inside the sweep).
    """
    out = []
    for batch in [1, 4, 8, 16, 32, 64, 128]:
        for max_ctx_len in [512, 1024, 2048, 4096, 8192]:
            for fp8_kv in [False, True]:
                out.append(dict(
                    batch=batch,
                    n_q_heads=n_q_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    max_ctx_len=max_ctx_len,
                    dtype=dtype,
                    fp8_kv=fp8_kv,
                ))
    return out


def gemm_decode_suite(
    N: int = 4096,
    K: int = 4096,
    dtype: str = "fp16",
    quant: str = "fp8",
) -> list[dict]:
    """The skinny-GEMM decode grid the autotune CLI sweeps."""
    return [
        dict(M=m, N=N, K=K, dtype=dtype, quant=quant)
        for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ]
