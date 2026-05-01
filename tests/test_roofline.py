"""Tests for RooflineCalculator."""

import math

import pytest

from kvforge.hardware import GPUSpec
from kvforge.optimizer.roofline import RooflineCalculator


@pytest.fixture
def fake_gpu() -> GPUSpec:
    # Fake GPU: 100 TFLOPS FP16, 1000 GB/s. Ridge point at AI = 100.
    return GPUSpec(
        name="FakeGPU", arch="fake",
        peak_fp16_tflops=100.0, peak_fp32_tflops=50.0,
        peak_bw_gb_s=1000.0, sm_count=64,
    )


def test_roofline_classifies_low_ai_as_memory_bound(fake_gpu):
    calc = RooflineCalculator(fake_gpu)
    # AI = 1 FLOP/byte → memory-bound.
    result = calc.analyze(flops=1_000_000, bytes_moved=1_000_000, runtime_seconds=1e-6)
    assert result.bound == "memory"


def test_roofline_classifies_high_ai_as_compute_bound(fake_gpu):
    calc = RooflineCalculator(fake_gpu)
    # AI = 200 FLOP/byte → compute-bound (above ridge point of 100).
    result = calc.analyze(flops=1_000_000_000, bytes_moved=5_000_000, runtime_seconds=1e-5)
    assert result.bound == "compute"


def test_roofline_pct_peak_bounded_at_one(fake_gpu):
    calc = RooflineCalculator(fake_gpu)
    # Ridiculously fast runtime → pct should saturate at 1.0, not exceed it.
    result = calc.analyze(flops=1_000_000, bytes_moved=1_000_000, runtime_seconds=1e-12)
    assert 0 <= result.pct_of_peak <= 1.0


def test_roofline_rejects_zero_runtime(fake_gpu):
    calc = RooflineCalculator(fake_gpu)
    with pytest.raises(ValueError):
        calc.analyze(flops=100, bytes_moved=100, runtime_seconds=0.0)


def test_roofline_recommends_tier_2_for_low_pct_memory_bound(fake_gpu):
    calc = RooflineCalculator(fake_gpu)
    result = calc.analyze(flops=1_000, bytes_moved=10_000, runtime_seconds=1e-3)
    tiers = calc.recommend_tier(result)
    assert any("Tier 2" in t for t in tiers)
    assert any("Tier 1" in t for t in tiers)


def test_roofline_skips_recommendations_at_near_peak(fake_gpu):
    calc = RooflineCalculator(fake_gpu)
    # Configure so the kernel hits 90%+ of achievable.
    flops, bytes_m = 1_000_000, 100_000  # AI = 10 → memory-bound
    achievable_gflops = min(100_000, 1000 * 10)  # = 10000 GFLOPS
    # Find runtime such that measured = 0.9 * achievable.
    target_gflops = 0.9 * achievable_gflops
    runtime_s = flops / target_gflops / 1e9
    result = calc.analyze(flops, bytes_m, runtime_s)
    assert result.pct_of_peak >= 0.85
    tiers = calc.recommend_tier(result)
    assert "near-peak" in tiers[0]
