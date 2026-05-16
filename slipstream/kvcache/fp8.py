"""FP8 (E4M3) quantize/dequantize with per-group scales.

We use E4M3 (4-bit exponent, 3-bit mantissa) for KV storage because it has a
wider mantissa than E5M2 — closer to FP16's precision for the small-magnitude
post-softmax values typical in attention. Range is roughly ±448.

Scaling strategy:
  scale = max(|x|) / FP8_E4M3_MAX   (per scale-group)
  x_fp8 = saturate(x / scale).cast(e4m3)
  x_recovered = x_fp8.cast(fp32) * scale

Per-token-per-head scaling (group = a single (token, head) head_dim vector)
preserves accuracy on outlier tokens at the cost of one fp16 scale per
(token, head). That overhead is < 1% of total KV bytes for typical head_dim,
so it's a free lunch vs. the alternative of dropping precision on outliers.

These functions are CPU-runnable. Triton kernels do the same math inline.
"""

from __future__ import annotations

import torch

# E4M3 finite range. torch.finfo(torch.float8_e4m3fn).max in PyTorch >= 2.1.
# Hardcoded so this module works without importing torch for the constant.
FP8_E4M3_MAX: float = 448.0
FP8_E4M3_MIN_POSITIVE: float = 2.0 ** -9  # ~1.95e-3


def quantize_fp8(
    x: torch.Tensor,
    group_dim: int = -1,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to FP8 E4M3 with per-group scales.

    Parameters
    ----------
    x:
        Input tensor in any float dtype.
    group_dim:
        The dimension that is *reduced* to compute one scale. For per-token-
        per-head KV scaling, ``x`` has shape ``[..., head_dim]`` and
        ``group_dim=-1``.
    scale_dtype:
        Storage dtype for scales. FP16 is plenty of range — scales are
        always positive.

    Returns
    -------
    (x_fp8, scales) where ``scales`` has ``x.shape`` with ``group_dim``
    removed. Multiply back to dequantize.

    Notes
    -----
    Tiny tensors with all-zero groups get a unit scale to avoid div-by-zero,
    which means zero stays zero through the round trip. Adversarially large
    inputs saturate at ±FP8_E4M3_MAX rather than blowing up.
    """
    x_fp32 = x.to(torch.float32)
    absmax = x_fp32.abs().amax(dim=group_dim, keepdim=True)

    # All-zero groups would produce scale=0 and a divide-by-zero. Replace
    # those with scale=1 (so zeros round-trip to zeros). The fp16 scale
    # storage also has a floor (~6e-5 normal); cast back through fp32 for
    # the divide preserves precision even when scales hit subnormals.
    is_zero_group = absmax == 0
    scales_fp32 = torch.where(is_zero_group, torch.ones_like(absmax), absmax / FP8_E4M3_MAX)
    scales = scales_fp32.to(scale_dtype)

    scaled = (x_fp32 / scales.to(torch.float32)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    x_fp8 = scaled.to(torch.float8_e4m3fn)

    return x_fp8, scales.squeeze(group_dim)


def dequantize_fp8(
    x_fp8: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
    group_dim: int = -1,
) -> torch.Tensor:
    """Inverse of :func:`quantize_fp8`. ``scales`` lacks ``group_dim``.

    Cast fp8 → fp32 for the multiply, then down-cast at the end. This matches
    what the Triton kernels do inline (dequant happens in fp32 registers).
    """
    s = scales.to(torch.float32).unsqueeze(group_dim)
    return (x_fp8.to(torch.float32) * s).to(out_dtype)


def quant_dequant_roundtrip(x: torch.Tensor, group_dim: int = -1) -> torch.Tensor:
    """Quantize then dequantize. Used for testing and for offline KV warming."""
    x_fp8, scales = quantize_fp8(x, group_dim=group_dim)
    return dequantize_fp8(x_fp8, scales, out_dtype=x.dtype, group_dim=group_dim)
