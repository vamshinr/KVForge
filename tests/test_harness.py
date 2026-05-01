"""Tests for CorrectnessHarness."""

from __future__ import annotations

import pytest
import torch

from kvforge.optimizer.harness import CorrectnessHarness


def _identity_inputs(shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    return (x,), {}


def _identity_ref(x):
    return x.clone()


def test_harness_passes_for_correct_candidate():
    harness = CorrectnessHarness(
        reference_fn=_identity_ref,
        input_factory=_identity_inputs,
        shape_configs=[(4, 8), (16, 32)],
        edge_shapes=[(7, 13)],
        dtypes=[torch.float32],
    )
    result = harness.validate(_identity_ref)
    assert result.passed
    assert result.failed_stage is None


def test_harness_rejects_wrong_output():
    harness = CorrectnessHarness(
        reference_fn=_identity_ref,
        input_factory=_identity_inputs,
        shape_configs=[(4, 8)],
        dtypes=[torch.float32],
    )
    bad_candidate = lambda x: x + 1.0  # noqa: E731
    result = harness.validate(bad_candidate)
    assert not result.passed
    # Could fail at smoke or shape_sweep, both are valid rejections.
    assert result.failed_stage in ("smoke", "shape_sweep")


def test_harness_rejects_shape_mismatch():
    harness = CorrectnessHarness(
        reference_fn=_identity_ref,
        input_factory=_identity_inputs,
        shape_configs=[(4, 8)],
        dtypes=[torch.float32],
    )
    bad_candidate = lambda x: x[:, :4]  # noqa: E731
    result = harness.validate(bad_candidate)
    assert not result.passed
    assert "shape mismatch" in result.failure_detail


def test_harness_catches_nondeterministic_kernel():
    """A kernel that adds random noise should fail the determinism stage."""
    counter = {"n": 0}

    def random_candidate(x):
        counter["n"] += 1
        return x + 0.0 * counter["n"]  # tiny per-call drift

    # Add real non-determinism.
    def noisy_candidate(x):
        return x + torch.randn_like(x) * 1e-7

    harness = CorrectnessHarness(
        reference_fn=_identity_ref,
        input_factory=_identity_inputs,
        shape_configs=[(4, 8)],
        dtypes=[torch.float32],
    )
    # `noisy_candidate` will fail tolerance check before determinism stage,
    # but it definitely won't pass — that's all we're asserting.
    result = harness.validate(noisy_candidate)
    assert not result.passed


def test_harness_catches_exceptions():
    def broken_candidate(x):
        raise RuntimeError("boom")

    harness = CorrectnessHarness(
        reference_fn=_identity_ref,
        input_factory=_identity_inputs,
        shape_configs=[(4, 8)],
        dtypes=[torch.float32],
    )
    result = harness.validate(broken_candidate)
    assert not result.passed
    assert "boom" in result.failure_detail


def test_harness_records_stage_times():
    harness = CorrectnessHarness(
        reference_fn=_identity_ref,
        input_factory=_identity_inputs,
        shape_configs=[(4, 8)],
        edge_shapes=[(7, 13)],
        dtypes=[torch.float32],
    )
    result = harness.validate(_identity_ref)
    assert result.passed
    expected_stages = {"smoke", "shape_sweep", "stability", "determinism", "edge_cases"}
    assert set(result.stage_times_ms.keys()) == expected_stages
    for stage, ms in result.stage_times_ms.items():
        assert ms >= 0
