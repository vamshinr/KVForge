"""Tests for the FP8 GEMM reference and its quantization helpers."""

from __future__ import annotations

import pytest
import torch

from slipstream.gemm.reference import (
    fp16_gemm_reference,
    fp8_gemm_reference,
    quantize_activation_per_token,
    quantize_weight_per_channel,
)


@pytest.mark.parametrize("M,N,K", [
    (1, 4096, 4096),     # decode batch=1
    (8, 4096, 4096),     # small decode batch
    (32, 4096, 4096),    # medium decode batch
    (128, 4096, 4096),   # large decode batch
])
def test_fp8_gemm_close_to_fp32_reference(M, N, K):
    """End-to-end: quant inputs → fp8 gemm → dequant → close to fp32 truth."""
    torch.manual_seed(0)
    a_hp = torch.randn(M, K, dtype=torch.float16)
    w_hp = torch.randn(K, N, dtype=torch.float16) * 0.02   # weights are smaller

    # Ground truth: fp32 matmul of the original high-precision tensors.
    truth = fp16_gemm_reference(a_hp, w_hp, out_dtype=torch.float32)

    a_fp8, sa = quantize_activation_per_token(a_hp)
    w_fp8, sb = quantize_weight_per_channel(w_hp)
    out = fp8_gemm_reference(a_fp8, w_fp8, sa, sb, out_dtype=torch.float32)

    # FP8 introduces ~3-bit-mantissa-equivalent noise per element of the
    # K-dim sum. Per-element relative error is a bad metric (truth has lots
    # of near-zero entries from sum-of-Gaussians cancellation). Two better
    # checks:
    #   1. Total RMS error is small relative to RMS of truth.
    #   2. Cosine similarity is near 1.
    err = (out - truth).flatten()
    rms_err = err.pow(2).mean().sqrt()
    rms_truth = truth.flatten().pow(2).mean().sqrt()
    rel_rms = (rms_err / rms_truth).item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten().unsqueeze(0), truth.flatten().unsqueeze(0)
    ).item()
    # Per-channel weight quant + per-token activation quant on Gaussian inputs
    # at K=4096 lands well under 5% RMS error and > 0.999 cosine.
    assert rel_rms < 0.05, f"M={M} N={N} K={K}: rel RMS error {rel_rms:.4f} too high"
    assert cos > 0.999, f"M={M} N={N} K={K}: cosine sim {cos:.6f} too low"


def test_fp8_gemm_scale_a_can_be_scalar():
    a = torch.randn(4, 8, dtype=torch.float16)
    w = torch.randn(8, 16, dtype=torch.float16)
    a_fp8, sa = quantize_activation_per_token(a)
    w_fp8, sb = quantize_weight_per_channel(w)

    # Per-token scale (rank-1)
    out_pt = fp8_gemm_reference(a_fp8, w_fp8, sa, sb)
    # Scalar scale (degenerate per-tensor): use a single mean scale instead.
    out_scalar = fp8_gemm_reference(a_fp8, w_fp8, sa.mean(), sb)

    # Both should be finite; numerically different but in the same ballpark.
    assert torch.isfinite(out_pt).all()
    assert torch.isfinite(out_scalar).all()


def test_weight_quantization_idempotent_after_quant():
    """A weight that's already representable in FP8 should round-trip exactly."""
    w = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float16)
    w_fp8, sb = quantize_weight_per_channel(w)
    w_back = (w_fp8.to(torch.float32) * sb.to(torch.float32))
    assert torch.allclose(w_back, w.to(torch.float32), atol=1e-3, rtol=1e-2)
