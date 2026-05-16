"""Eager FP8 GEMM reference.

Math (matches the hipBLASLt FP8 API):

    A:        [M, K]  fp8_e4m3
    B:        [K, N]  fp8_e4m3
    scale_a:  [M, 1] or scalar  (per-token A scaling)
    scale_b:  [1, N]            (per-channel B scaling)
    D = (A_fp32 @ B_fp32) * scale_a * scale_b      → fp16 / bf16

scale_a per-token is the standard activation scaling for transformer
decode: each row of the activation matrix gets its own scale. scale_b
per-channel matches weight-only quantization conventions; the weight
matrix is quantized once offline and the per-column scale is published.

This is the oracle. The Triton template will reproduce these semantics
exactly inside an MFMA tile loop.
"""

from __future__ import annotations

import torch

from slipstream.kvcache.fp8 import FP8_E4M3_MAX


def fp8_gemm_reference(
    a_fp8: torch.Tensor,         # [M, K] fp8_e4m3fn
    b_fp8: torch.Tensor,         # [K, N] fp8_e4m3fn
    scale_a: torch.Tensor,       # [M] or scalar  (per-token or per-tensor)
    scale_b: torch.Tensor,       # [N]            (per-channel along N)
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Compute ``D = (A_fp32 @ B_fp32) * scale_a[:, None] * scale_b[None, :]``.

    Per-token scale_a is shape ``[M]`` (one scalar per query token);
    a 0-D scalar is broadcast as per-tensor. scale_b is always ``[N]``.
    """
    assert a_fp8.dtype == torch.float8_e4m3fn, f"A must be fp8_e4m3fn, got {a_fp8.dtype}"
    assert b_fp8.dtype == torch.float8_e4m3fn, f"B must be fp8_e4m3fn, got {b_fp8.dtype}"
    assert a_fp8.ndim == 2 and b_fp8.ndim == 2
    M, K = a_fp8.shape
    K2, N = b_fp8.shape
    assert K == K2, f"K mismatch: A={K} B={K2}"

    a_fp32 = a_fp8.to(torch.float32)
    b_fp32 = b_fp8.to(torch.float32)
    raw = a_fp32 @ b_fp32                              # [M, N]

    sa = scale_a.to(torch.float32).reshape(-1, 1) if scale_a.numel() > 1 \
         else scale_a.to(torch.float32)
    sb = scale_b.to(torch.float32).reshape(1, -1)
    return (raw * sa * sb).to(out_dtype)


def fp16_gemm_reference(
    a: torch.Tensor,             # [M, K] fp16/bf16
    b: torch.Tensor,             # [K, N] fp16/bf16
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Vanilla FP16/BF16 GEMM with FP32 accumulate. Reference for the few
    decode-step matmuls we don't quantize.

    Done in fp32 internally to match what tensor cores do with FP32 accumulate.
    """
    out_dtype = out_dtype or a.dtype
    return (a.to(torch.float32) @ b.to(torch.float32)).to(out_dtype)


# ---------- Quantization helpers (activation side) ----------


def quantize_activation_per_token(
    x: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (per-token) FP8 quantization for activations.

    Returns ``(x_fp8, scale_a)`` where ``scale_a`` shape is ``[M]``. Matches
    the input convention expected by :func:`fp8_gemm_reference`.

    Per-token scaling preserves the dynamic range of outlier tokens (long-tail
    activations after RMSNorm are common in LLM inference), at the cost of one
    fp16 scale per row. Negligible overhead vs the matmul itself.
    """
    assert x.ndim == 2
    x_fp32 = x.to(torch.float32)
    absmax = x_fp32.abs().amax(dim=-1, keepdim=True)
    is_zero = absmax == 0
    scales = torch.where(is_zero, torch.ones_like(absmax), absmax / FP8_E4M3_MAX)

    scaled = (x_fp32 / scales).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = scaled.to(out_dtype)
    return x_fp8, scales.squeeze(-1).to(torch.float16)


def quantize_weight_per_channel(
    w: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column (per-output-channel) FP8 quantization for weights.

    ``w`` shape ``[K, N]``. Returns ``(w_fp8, scale_b)`` with ``scale_b``
    shape ``[N]``. Per-channel is the SoTA quantization granularity for
    weight-only / weight+activation FP8 — coarser than per-element, but
    captures the actual per-output-feature dynamic range.

    Done once offline; the cached weight is then used for all subsequent
    GEMMs of that linear layer.
    """
    assert w.ndim == 2
    w_fp32 = w.to(torch.float32)
    absmax = w_fp32.abs().amax(dim=0, keepdim=True)
    is_zero = absmax == 0
    scales = torch.where(is_zero, torch.ones_like(absmax), absmax / FP8_E4M3_MAX)

    scaled = (w_fp32 / scales).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    w_fp8 = scaled.to(out_dtype)
    return w_fp8, scales.squeeze(0).to(torch.float16)
