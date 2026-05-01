"""Tests for AmdahlRanker."""

from __future__ import annotations

import math

import pytest

from kvforge.profiler.amdahl import AmdahlRanker, KernelEntry
from kvforge.profiler.classify import OpType


@pytest.fixture
def ranker() -> AmdahlRanker:
    return AmdahlRanker(projections=(2.0, 5.0))


def test_amdahl_law_formula() -> None:
    """The classic Amdahl example: 50% kernel × 2x speedup → 4/3x total."""
    e = KernelEntry(name="k", op_type=OpType.OTHER, total_us=100.0, call_count=1, fraction=0.5)
    assert math.isclose(e.project(2.0), 4 / 3, rel_tol=1e-9)


def test_amdahl_at_unity_speedup_yields_no_change() -> None:
    e = KernelEntry(name="k", op_type=OpType.OTHER, total_us=10.0, call_count=1, fraction=0.6)
    assert math.isclose(e.project(1.0), 1.0)


def test_amdahl_unbounded_speedup_capped_by_serial_fraction() -> None:
    """As local speedup → ∞, total speedup → 1 / (1 - f)."""
    e = KernelEntry(name="k", op_type=OpType.OTHER, total_us=50.0, call_count=1, fraction=0.5)
    huge = e.project(1e9)
    assert math.isclose(huge, 2.0, rel_tol=1e-3)


def test_amdahl_rejects_nonpositive_speedup() -> None:
    e = KernelEntry(name="k", op_type=OpType.OTHER, total_us=10.0, call_count=1, fraction=0.5)
    with pytest.raises(ValueError):
        e.project(0.0)


def test_ranker_orders_by_descending_fraction(ranker: AmdahlRanker) -> None:
    times = {
        "small_kernel": (10.0, 1),
        "big_kernel": (60.0, 1),
        "medium_kernel": (30.0, 1),
    }
    entries = ranker.rank(times)
    assert [e.name for e in entries] == ["big_kernel", "medium_kernel", "small_kernel"]
    assert [e.rank for e in entries] == [1, 2, 3]


def test_ranker_fractions_sum_to_one(ranker: AmdahlRanker) -> None:
    times = {"a": (20.0, 1), "b": (30.0, 1), "c": (50.0, 1)}
    entries = ranker.rank(times)
    assert math.isclose(sum(e.fraction for e in entries), 1.0)


def test_ranker_handles_empty_input(ranker: AmdahlRanker) -> None:
    assert ranker.rank({}) == []


def test_ranker_handles_zero_total_time(ranker: AmdahlRanker) -> None:
    assert ranker.rank({"k": (0.0, 0)}) == []


def test_ranker_aggregation_collapses_same_op_type(ranker: AmdahlRanker) -> None:
    times = {
        "cublas_gemm_v1": (50.0, 1),
        "cublas_gemm_v2": (30.0, 1),
        "rmsnorm_kernel": (20.0, 1),
    }
    op_types = {
        "cublas_gemm_v1": OpType.MATMUL,
        "cublas_gemm_v2": OpType.MATMUL,
        "rmsnorm_kernel": OpType.RMSNORM,
    }
    entries = ranker.rank(times, op_types)
    aggregated = ranker.aggregate_by_op_type(entries)
    assert len(aggregated) == 2
    matmul = next(e for e in aggregated if e.op_type == OpType.MATMUL)
    assert matmul.total_us == 80.0
    assert math.isclose(matmul.fraction, 0.8)


def test_ranker_projections_are_populated(ranker: AmdahlRanker) -> None:
    entries = ranker.rank({"k": (100.0, 1)})
    assert 2.0 in entries[0].projections
    assert 5.0 in entries[0].projections
    assert entries[0].projections[2.0] > 1.0
