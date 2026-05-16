"""Tests for the autotune cache and sweep driver."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from slipstream.autotune.cache import AutotuneCache, CacheKey, TunedConfig
from slipstream.autotune.shape_buckets import (
    attention_decode_suite,
    bucket_attention_shape,
    bucket_gemm_shape,
    gemm_decode_suite,
)
from slipstream.autotune.sweep import sweep_configs


# ---------- Cache persistence ----------


def test_cache_roundtrip_persists_to_disk():
    with tempfile.TemporaryDirectory() as d:
        cache = AutotuneCache(arch="cpu", cache_dir=d)
        key = CacheKey("paged_attn", "b8_qh32_kvh8_d128_ctx2048_fp16_fp8")
        tuned = TunedConfig(
            config={"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
            measured_ms=0.123,
            triton_version="3.0.0",
        )
        cache.record(key, tuned)

        # Fresh handle, same dir — should see the persisted entry.
        cache2 = AutotuneCache(arch="cpu", cache_dir=d)
        got = cache2.lookup(key)
        assert got is not None
        assert got.config == tuned.config
        assert got.measured_ms == 0.123
        assert got.recorded_at != ""    # auto-stamped


def test_cache_returns_none_on_miss():
    with tempfile.TemporaryDirectory() as d:
        cache = AutotuneCache(arch="cpu", cache_dir=d)
        assert cache.lookup(CacheKey("not_there", "no_shape")) is None


def test_cache_overwrites_existing_entry():
    with tempfile.TemporaryDirectory() as d:
        cache = AutotuneCache(arch="cpu", cache_dir=d)
        key = CacheKey("k", "s")
        cache.record(key, TunedConfig(config={"x": 1}, measured_ms=2.0))
        cache.record(key, TunedConfig(config={"x": 2}, measured_ms=1.0))
        got = cache.lookup(key)
        assert got is not None and got.config == {"x": 2}


def test_cache_corrupted_file_falls_back_to_empty():
    with tempfile.TemporaryDirectory() as d:
        # Write garbage to the expected path.
        Path(d, "cpu.json").write_text("this is not json")
        cache = AutotuneCache(arch="cpu", cache_dir=d)
        assert cache.lookup(CacheKey("k", "s")) is None
        # Recording over it should work — corruption is recoverable.
        cache.record(CacheKey("k", "s"), TunedConfig(config={"a": 1}, measured_ms=0.5))
        assert cache.lookup(CacheKey("k", "s")) is not None


# ---------- Shape bucketing ----------


def test_bucket_attention_rounds_batch_and_ctx_to_grid():
    a = bucket_attention_shape(
        batch=33, n_q_heads=32, n_kv_heads=8, head_dim=128,
        max_ctx_len=2050, dtype="fp16", fp8_kv=True,
    )
    b = bucket_attention_shape(
        batch=32, n_q_heads=32, n_kv_heads=8, head_dim=128,
        max_ctx_len=2048, dtype="fp16", fp8_kv=True,
    )
    # batch 33 rounds up to 64 (next bucket); ctx 2050 rounds to 3072.
    # Either way, they should differ from the exact bucket — bucketing must
    # be functional but stable.
    assert b == "b32_qh32_kvh8_d128_ctx2048_fp16_fp8"
    assert "b32" not in a   # 33 didn't collapse onto 32
    assert "ctx2048" not in a   # 2050 didn't collapse onto 2048


def test_bucket_gemm_only_buckets_M():
    # N and K are model constants — exact.
    s = bucket_gemm_shape(M=33, N=4096, K=4096, dtype="fp16", quant="fp8")
    assert "n4096" in s
    assert "k4096" in s
    # M=33 rounds to 64 (next bucket).
    assert "m64" in s


def test_decode_suite_covers_realistic_shapes():
    a_suite = attention_decode_suite()
    assert any(c["batch"] == 32 and c["max_ctx_len"] == 2048 and c["fp8_kv"] for c in a_suite)
    g_suite = gemm_decode_suite()
    assert any(c["M"] == 1 for c in g_suite)
    assert any(c["M"] == 128 for c in g_suite)


# ---------- Sweep driver ----------


def test_sweep_picks_fastest_passing_config():
    candidates = [
        {"BLOCK_M": 32, "num_warps": 2},   # slowest
        {"BLOCK_M": 64, "num_warps": 4},   # fastest
        {"BLOCK_M": 128, "num_warps": 8},  # middle
    ]
    times = {
        (32, 2): 3.0,
        (64, 4): 1.0,
        (128, 8): 2.0,
    }

    def run(cfg):
        return times[(cfg["BLOCK_M"], cfg["num_warps"])]

    result = sweep_configs(
        kernel_id="fake", shape_bucket="s",
        candidates=candidates, run=run,
    )
    assert result.succeeded()
    assert result.winner.config == {"BLOCK_M": 64, "num_warps": 4}
    assert result.winner.measured_ms == 1.0
    assert result.n_candidates == 3
    assert result.n_failed == 0


def test_sweep_skips_failing_configs():
    def run(cfg):
        if cfg["BLOCK_M"] == 128:
            raise RuntimeError("out of resources")
        return cfg["BLOCK_M"] * 0.01

    result = sweep_configs(
        kernel_id="fake", shape_bucket="s",
        candidates=[{"BLOCK_M": 64}, {"BLOCK_M": 128}, {"BLOCK_M": 256}],
        run=run,
    )
    assert result.succeeded()
    assert result.winner.config == {"BLOCK_M": 64}
    assert result.n_failed == 1


def test_sweep_returns_no_winner_when_all_fail():
    def run(cfg):
        raise RuntimeError("nope")

    result = sweep_configs(
        kernel_id="fake", shape_bucket="s",
        candidates=[{"BLOCK_M": 32}, {"BLOCK_M": 64}],
        run=run,
    )
    assert not result.succeeded()
    assert result.winner is None
    assert result.n_failed == 2


def test_sweep_honors_verify_gate():
    def run(cfg):
        return 1.0

    def verify(cfg):
        return cfg["BLOCK_M"] != 64   # 64 is "incorrect"

    result = sweep_configs(
        kernel_id="fake", shape_bucket="s",
        candidates=[{"BLOCK_M": 32}, {"BLOCK_M": 64}, {"BLOCK_M": 128}],
        run=run, verify=verify,
    )
    assert result.succeeded()
    assert result.winner.config["BLOCK_M"] in {32, 128}
    assert result.n_failed == 1


def test_sweep_writes_winner_to_cache():
    with tempfile.TemporaryDirectory() as d:
        cache = AutotuneCache(arch="cpu", cache_dir=d)
        sweep_configs(
            kernel_id="k1", shape_bucket="s1",
            candidates=[{"x": 1}, {"x": 2}],
            run=lambda cfg: float(cfg["x"]),
            cache=cache,
        )
        got = cache.lookup(CacheKey("k1", "s1"))
        assert got is not None
        assert got.config == {"x": 1}
