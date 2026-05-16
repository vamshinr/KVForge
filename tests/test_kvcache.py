"""Tests for the paged KV cache and FP8 quantization."""

from __future__ import annotations

import pytest
import torch

from slipstream.kvcache.fp8 import (
    FP8_E4M3_MAX,
    dequantize_fp8,
    quant_dequant_roundtrip,
    quantize_fp8,
)
from slipstream.kvcache.paged import BlockTable, PagedKVCache, SequenceState


# ---------- FP8 round-trip ----------


@pytest.mark.parametrize("shape", [(4, 8, 128), (1, 1, 64), (16, 4, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fp8_roundtrip_small_relative_error(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype)
    rec = quant_dequant_roundtrip(x, group_dim=-1)

    # Per-group scaling: relative error should be ~ 1 / 2^3 = 0.125 worst case
    # (E4M3 has 3 mantissa bits + implicit leading 1). Real error is much
    # smaller because the scale puts the largest element exactly at FP8_MAX.
    rel = (rec.to(torch.float32) - x.to(torch.float32)).abs() / x.abs().clamp_min(1e-3)
    assert rel.mean() < 0.05, f"avg relative error {rel.mean().item()} too high"
    assert rel.max() < 0.15, f"max relative error {rel.max().item()} too high"


def test_fp8_handles_all_zero_group():
    x = torch.zeros(2, 1, 128)
    rec = quant_dequant_roundtrip(x)
    assert torch.equal(rec, x)


def test_fp8_saturates_at_large_values():
    # E4M3 max is ~448. Quantize a tensor where one outlier dominates,
    # then check the small values still survive after scale recovery.
    x = torch.tensor([[1.0, 2.0, 3.0, 1e6]], dtype=torch.float32)
    x_q, s = quantize_fp8(x, group_dim=-1)
    rec = dequantize_fp8(x_q, s, out_dtype=torch.float32, group_dim=-1)
    # The outlier dominates the scale; small values now occupy < 1% of FP8
    # range and lose precision. That's expected — this test just confirms no
    # NaN/Inf and the outlier is approximately recovered.
    assert torch.isfinite(rec).all()
    assert abs(rec[0, 3].item() - 1e6) / 1e6 < 0.05


# ---------- Block table ----------


def test_block_table_alloc_free_cycle():
    bt = BlockTable(num_blocks=8)
    assert bt.num_free == 8

    a = bt.allocate(3)
    assert bt.num_free == 5
    assert len(set(a)) == 3       # distinct ids

    b = bt.allocate(2)
    assert bt.num_free == 3
    assert set(a).isdisjoint(set(b))

    bt.free(a)
    assert bt.num_free == 6

    # After free, allocations should reuse — LIFO means the most recently
    # freed block comes back first, which is good for cache locality.
    c = bt.allocate(1)
    assert c[0] in a


def test_block_table_oversubscription_raises():
    bt = BlockTable(num_blocks=2)
    bt.allocate(2)
    with pytest.raises(RuntimeError, match="out of KV cache blocks"):
        bt.allocate(1)


# ---------- Paged cache: append + gather, FP16 ----------


@pytest.fixture
def small_cache_fp16():
    return PagedKVCache(
        num_blocks=16, block_size=4, n_kv_heads=2, head_dim=8,
        kv_dtype=torch.float16, device="cpu",
    )


def test_paged_cache_append_and_gather_roundtrip(small_cache_fp16):
    cache = small_cache_fp16
    seq = SequenceState(seq_id=0)

    # Append 10 tokens (forces 3 blocks of size 4).
    k = torch.randn(10, 2, 8, dtype=torch.float16)
    v = torch.randn(10, 2, 8, dtype=torch.float16)
    cache.append(seq, k, v)

    assert seq.length == 10
    assert len(seq.block_ids) == 3   # ceil(10/4)

    k_back, v_back = cache.gather_sequence(seq, out_dtype=torch.float16)
    assert k_back.shape == (10, 2, 8)
    assert torch.allclose(k_back, k, atol=1e-3)
    assert torch.allclose(v_back, v, atol=1e-3)


def test_paged_cache_free_returns_blocks(small_cache_fp16):
    cache = small_cache_fp16
    seq = SequenceState(seq_id=0)
    cache.append(seq, torch.zeros(8, 2, 8, dtype=torch.float16),
                       torch.zeros(8, 2, 8, dtype=torch.float16))
    assert cache.block_table.num_free == 16 - 2

    cache.free(seq)
    assert cache.block_table.num_free == 16
    assert seq.length == 0
    assert seq.block_ids == []


# ---------- Paged cache: FP8 ----------


def test_paged_cache_fp8_roundtrip_close_enough():
    cache = PagedKVCache(
        num_blocks=8, block_size=4, n_kv_heads=2, head_dim=64,
        kv_dtype=torch.float8_e4m3fn, device="cpu",
    )
    seq = SequenceState(seq_id=0)

    torch.manual_seed(42)
    k = torch.randn(6, 2, 64, dtype=torch.float16)
    v = torch.randn(6, 2, 64, dtype=torch.float16)
    cache.append(seq, k, v)

    k_back, v_back = cache.gather_sequence(seq, out_dtype=torch.float16)
    # FP8 round-trip — looser tolerance than FP16-storage case.
    err_k = (k_back.to(torch.float32) - k.to(torch.float32)).abs().mean().item()
    err_v = (v_back.to(torch.float32) - v.to(torch.float32)).abs().mean().item()
    assert err_k < 0.05, f"mean fp8 KV cache error too high: K={err_k}"
    assert err_v < 0.05, f"mean fp8 KV cache error too high: V={err_v}"


def test_paged_cache_per_block_bytes_includes_scales():
    fp16 = PagedKVCache(num_blocks=1, block_size=16, n_kv_heads=8, head_dim=128,
                        kv_dtype=torch.float16)
    fp8 = PagedKVCache(num_blocks=1, block_size=16, n_kv_heads=8, head_dim=128,
                       kv_dtype=torch.float8_e4m3fn)
    # FP8 has half the K/V bytes but adds scale storage.
    # K+V: 2 * 16 * 8 * 128 * (2 or 1) bytes; scales: 2 * 16 * 8 * 2 bytes.
    assert fp16.per_block_bytes() == 2 * 16 * 8 * 128 * 2
    assert fp8.per_block_bytes() == 2 * 16 * 8 * 128 * 1 + 2 * 16 * 8 * 2
    # FP8 cache is roughly half the size.
    assert fp8.per_block_bytes() < fp16.per_block_bytes() * 0.55


def test_sequence_state_slot_lookup():
    seq = SequenceState(seq_id=0)
    seq.block_ids = [10, 20, 30]
    seq.length = 9
    # position 0 → (10, 0), position 4 → (20, 0), position 8 → (30, 0)
    assert seq.slot_for(0, block_size=4) == (10, 0)
    assert seq.slot_for(4, block_size=4) == (20, 0)
    assert seq.slot_for(8, block_size=4) == (30, 0)
    with pytest.raises(IndexError):
        seq.slot_for(9, block_size=4)
