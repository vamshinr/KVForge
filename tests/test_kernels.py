"""Numerical correctness tests for KVForge kernels.

These tests run on CPU (eager fallback) by default and on GPU when CUDA is
available. They are dtype-aware: fp16/bf16 use looser tolerances than fp32.
"""

from __future__ import annotations

import pytest
import torch

from kvforge.kernels.rmsnorm import rmsnorm, rmsnorm_reference
from kvforge.kernels.rope import rope, rope_reference
from kvforge.kernels.softmax import softmax, softmax_reference


def _tol(dtype: torch.dtype) -> tuple[float, float]:
    return {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (1e-2, 1e-3),
        torch.bfloat16: (2e-2, 1e-2),
    }.get(dtype, (1e-4, 1e-5))


# ---------- RMSNorm ----------


@pytest.mark.parametrize("shape", [(4, 2048), (16, 4096), (1, 2048), (32, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_rmsnorm_matches_reference_cpu(shape, dtype, device):
    if device.type != "cpu":
        pytest.skip("CPU-only test")
    x = torch.randn(*shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype)
    out_ref = rmsnorm_reference(x, w)
    out = rmsnorm(x, w)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(4, 2048), (16, 4096), (32, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_matches_reference_gpu(shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device)
    out_ref = rmsnorm_reference(x, w)
    out = rmsnorm(x, w)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out.float(), out_ref.float(), atol=atol, rtol=rtol)


@pytest.mark.gpu
def test_rmsnorm_handles_non_power_of_two_shape(device):
    """Edge case: non-pow2 hidden dim should still produce correct output."""
    x = torch.randn(8, 1023, dtype=torch.float32, device=device)
    w = torch.randn(1023, dtype=torch.float32, device=device)
    out = rmsnorm(x, w)
    out_ref = rmsnorm_reference(x, w)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-5)


@pytest.mark.gpu
def test_rmsnorm_is_deterministic(device):
    """Three runs must produce bitwise identical outputs."""
    x = torch.randn(4, 2048, dtype=torch.float32, device=device)
    w = torch.randn(2048, dtype=torch.float32, device=device)
    out1 = rmsnorm(x, w).clone()
    out2 = rmsnorm(x, w).clone()
    out3 = rmsnorm(x, w).clone()
    assert torch.equal(out1, out2)
    assert torch.equal(out2, out3)


# ---------- RoPE ----------


@pytest.mark.parametrize("shape", [(1, 4, 256, 64), (2, 8, 128, 64)])
def test_rope_matches_reference_cpu(shape, device):
    if device.type != "cpu":
        pytest.skip("CPU-only test")
    B, H, S, D = shape
    x = torch.randn(*shape, dtype=torch.float32)
    angles = torch.arange(S).float()[:, None] * torch.arange(0, D, 2).float()
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    out = rope(x, cos, sin)
    out_ref = rope_reference(x, cos, sin)
    assert torch.allclose(out, out_ref, atol=1e-4, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 8, 1024, 128), (2, 4, 512, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rope_matches_reference_gpu(shape, dtype, device):
    B, H, S, D = shape
    x = torch.randn(*shape, dtype=dtype, device=device)
    angles = torch.arange(S, device=device).float()[:, None] * torch.arange(0, D, 2, device=device).float()
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)
    out = rope(x, cos, sin)
    out_ref = rope_reference(x, cos, sin)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out.float(), out_ref.float(), atol=atol, rtol=rtol)


# ---------- Softmax ----------


@pytest.mark.parametrize("shape", [(4, 2048), (16, 4096), (1, 1024)])
def test_softmax_matches_reference_cpu(shape, device):
    if device.type != "cpu":
        pytest.skip("CPU-only test")
    x = torch.randn(*shape, dtype=torch.float32)
    out = softmax(x, dim=-1)
    out_ref = softmax_reference(x, dim=-1)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(4, 2048), (16, 4096), (32, 8192)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_softmax_matches_reference_gpu(shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    out = softmax(x, dim=-1)
    out_ref = softmax_reference(x, dim=-1)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out.float(), out_ref.float(), atol=atol, rtol=rtol)


@pytest.mark.gpu
def test_softmax_rows_sum_to_one(device):
    """Sanity check: each row of output should sum to ~1.0."""
    x = torch.randn(16, 4096, dtype=torch.float32, device=device)
    out = softmax(x, dim=-1)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.gpu
def test_softmax_handles_large_values(device):
    """Numerical stability: large inputs should not overflow."""
    x = torch.randn(4, 2048, dtype=torch.float32, device=device) * 100
    out = softmax(x, dim=-1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
