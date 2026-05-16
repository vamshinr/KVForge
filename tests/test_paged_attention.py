"""Tests for the paged attention reference impl."""

from __future__ import annotations

import math

import pytest
import torch

from slipstream.attention.reference import (
    paged_attention_reference,
    paged_attention_reference_prefill,
)
from slipstream.kvcache.paged import PagedKVCache, SequenceState


def _make_cache(n_kv_heads: int, head_dim: int, kv_dtype: torch.dtype = torch.float16) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=32, block_size=16,
        n_kv_heads=n_kv_heads, head_dim=head_dim,
        kv_dtype=kv_dtype, device="cpu",
    )


def _populate(
    cache: PagedKVCache, seq_lengths: list[int], seed: int = 0,
) -> list[SequenceState]:
    g = torch.Generator().manual_seed(seed)
    seqs: list[SequenceState] = []
    for sid, n in enumerate(seq_lengths):
        seq = SequenceState(seq_id=sid)
        k = torch.randn(n, cache.n_kv_heads, cache.head_dim, generator=g, dtype=torch.float16)
        v = torch.randn(n, cache.n_kv_heads, cache.head_dim, generator=g, dtype=torch.float16)
        cache.append(seq, k, v)
        seqs.append(seq)
    return seqs


def _naive_decode_attention(
    q: torch.Tensor,                # [B, Hq, D]
    k_full: list[torch.Tensor],     # per-seq: [L, Hkv, D]
    v_full: list[torch.Tensor],
    n_kv_heads: int,
) -> torch.Tensor:
    """Brute-force decode attention with no cache abstraction."""
    B, Hq, D = q.shape
    group_size = Hq // n_kv_heads
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    for b in range(B):
        k = k_full[b].repeat_interleave(group_size, dim=1).to(torch.float32)   # [L, Hq, D]
        v = v_full[b].repeat_interleave(group_size, dim=1).to(torch.float32)
        qb = q[b].to(torch.float32)
        scores = torch.einsum("hd,lhd->hl", qb, k) * scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hl,lhd->hd", probs, v).to(q.dtype)
    return out


# ---------- FP16 paged decode matches the brute-force impl ----------


@pytest.mark.parametrize("n_kv_heads,n_q_heads,head_dim", [
    (2, 8, 64),     # GQA group size 4 (Llama-3-8B style)
    (1, 1, 32),     # MHA degenerate
    (4, 4, 128),    # MHA full
    (2, 16, 128),   # large GQA group
])
@pytest.mark.parametrize("seq_lengths", [
    [1],
    [10, 25],
    [16, 32, 1],
])
def test_paged_attention_reference_matches_brute_force_fp16(
    n_kv_heads, n_q_heads, head_dim, seq_lengths,
):
    torch.manual_seed(123)
    cache = _make_cache(n_kv_heads, head_dim, kv_dtype=torch.float16)
    seqs = _populate(cache, seq_lengths)

    q = torch.randn(len(seqs), n_q_heads, head_dim, dtype=torch.float16)
    out = paged_attention_reference(q, cache, seqs)

    # Brute-force using the same K/V that the cache stores (re-gather to
    # account for the FP16 storage round-trip, which is bit-identical here).
    k_full = [cache.gather_sequence(s, out_dtype=torch.float16)[0] for s in seqs]
    v_full = [cache.gather_sequence(s, out_dtype=torch.float16)[1] for s in seqs]
    expected = _naive_decode_attention(q, k_full, v_full, n_kv_heads)

    # FP16-stored, FP32-computed → tight match.
    assert torch.allclose(out.to(torch.float32), expected.to(torch.float32),
                          atol=5e-3, rtol=5e-3), \
        f"diff max={ (out-expected).abs().max().item() }"


# ---------- FP8 KV path produces sane output (looser tolerance) ----------


def test_paged_attention_fp8_kv_close_to_fp16_kv():
    torch.manual_seed(7)
    cache_16 = _make_cache(2, 64, kv_dtype=torch.float16)
    cache_8  = _make_cache(2, 64, kv_dtype=torch.float8_e4m3fn)

    seq_lengths = [12, 30]
    seqs_16 = _populate(cache_16, seq_lengths, seed=99)
    seqs_8  = _populate(cache_8,  seq_lengths, seed=99)

    q = torch.randn(2, 8, 64, dtype=torch.float16)

    out_16 = paged_attention_reference(q, cache_16, seqs_16)
    out_8  = paged_attention_reference(q, cache_8,  seqs_8)

    # FP8 KV adds quantization noise. Per-token-per-head scaling keeps the
    # delta small in practice; 5% is a generous bound but well above
    # quantization floor for these shapes.
    rel_err = (out_8.to(torch.float32) - out_16.to(torch.float32)).abs().mean() \
              / out_16.to(torch.float32).abs().mean()
    assert rel_err.item() < 0.05, f"FP8 KV rel error {rel_err.item()} too high"


# ---------- Prefill reference: causal mask is applied ----------


def test_prefill_causal_mask_zeros_future():
    """Two-token prefill: the first token's output must not depend on the
    second token's K/V (causal).
    """
    torch.manual_seed(0)
    cache_a = _make_cache(1, 16, kv_dtype=torch.float16)
    cache_b = _make_cache(1, 16, kv_dtype=torch.float16)

    seq_a = SequenceState(seq_id=0)
    seq_b = SequenceState(seq_id=0)

    k1 = torch.randn(1, 1, 16, dtype=torch.float16)
    v1 = torch.randn(1, 1, 16, dtype=torch.float16)
    k2_a = torch.randn(1, 1, 16, dtype=torch.float16)
    v2_a = torch.randn(1, 1, 16, dtype=torch.float16)
    k2_b = torch.randn(1, 1, 16, dtype=torch.float16)   # different second token
    v2_b = torch.randn(1, 1, 16, dtype=torch.float16)

    cache_a.append(seq_a, torch.cat([k1, k2_a]), torch.cat([v1, v2_a]))
    cache_b.append(seq_b, torch.cat([k1, k2_b]), torch.cat([v1, v2_b]))

    q = torch.randn(2, 1, 16, dtype=torch.float16)
    out_a = paged_attention_reference_prefill(q, cache_a, [seq_a], q_lens=[2])
    out_b = paged_attention_reference_prefill(q, cache_b, [seq_b], q_lens=[2])

    # First query token (causal) should be identical between A and B since
    # only the second token differs.
    assert torch.allclose(out_a[0], out_b[0], atol=1e-3, rtol=1e-3), \
        "first prefill token output changed when only future K/V differed — causal mask broken"

    # Second token should differ (it attends to itself, which differs).
    assert not torch.allclose(out_a[1], out_b[1], atol=1e-3, rtol=1e-3)
