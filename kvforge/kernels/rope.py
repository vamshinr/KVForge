"""Rotary Position Embeddings (RoPE): optimized Triton kernel + eager reference.

RoPE rotates pairs of dimensions in the head dim by a position-dependent
angle:

    [x_even, x_odd] -> [x_even * cos - x_odd * sin,
                       x_odd  * cos + x_even * sin]

In the eager implementation this requires:
  1. slicing odd/even dims (two strided reads),
  2. stacking into a new tensor (extra allocation),
  3. multiply-add against precomputed cos/sin tables.

The Triton kernel does all three steps in a single pass over each
[batch, head, seq, head_dim] vector, with cos/sin loaded from a small cached
table.

This kernel is interesting because it's *barely* memory-bound — the AI is
~2 FLOPs/byte — so block-size tuning has outsized impact.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------- Eager reference ----------


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Eager RoPE matching `kvforge.models.tinyllama.apply_rope`."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotated * sin


# ---------- Triton kernel ----------


if _HAS_TRITON:

    @triton.jit
    def _rope_fwd_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        y_ptr,
        x_stride_b, x_stride_h, x_stride_s, x_stride_d,
        n_seq,
        head_dim,
        BLOCK_HEAD: tl.constexpr,
    ):
        """One program per (batch, head, seq) triple. Rotates one head_dim vector.

        head_dim is processed as a single contiguous block with size BLOCK_HEAD
        (must be >= head_dim, power of two). For typical Llama head_dim=64 or
        128 this fits comfortably in registers.
        """
        b = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.program_id(2)

        # Compute the base offset for this (batch, head, seq) row.
        base = b * x_stride_b + h * x_stride_h + s * x_stride_s

        # Load the head_dim vector with even/odd interleave.
        offsets = tl.arange(0, BLOCK_HEAD)
        mask = offsets < head_dim

        # Even and odd indices.
        even_offsets = (offsets // 2) * 2
        odd_offsets = even_offsets + 1
        even_mask = (even_offsets < head_dim) & (offsets < head_dim) & ((offsets % 2) == 0)
        odd_mask = (odd_offsets < head_dim) & (offsets < head_dim) & ((offsets % 2) == 1)

        x_even = tl.load(x_ptr + base + even_offsets * x_stride_d, mask=mask, other=0.0)
        x_odd = tl.load(x_ptr + base + odd_offsets * x_stride_d, mask=mask, other=0.0)

        # Load cos/sin for this seq position.
        cos = tl.load(cos_ptr + s * head_dim + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + s * head_dim + offsets, mask=mask, other=0.0)

        # Apply rotation. The "even" lanes get x*cos - x_odd*sin pattern;
        # "odd" lanes get x*cos + x_even*sin. Branchless via select on parity.
        is_even = (offsets % 2) == 0
        out = tl.where(
            is_even,
            x_even * cos - x_odd * sin,
            x_odd * cos + x_even * sin,
        )

        tl.store(y_ptr + base + offsets * x_stride_d, out, mask=mask)


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Public RoPE forward.

    Parameters
    ----------
    x: shape ``[batch, n_heads, seq, head_dim]``.
    cos / sin: shape ``[seq, head_dim]`` — precomputed tables.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return rope_reference(x, cos, sin)

    B, H, S, D = x.shape
    if D > 256:
        # head_dim > 256 is unusual; fall back rather than risk register spills.
        return rope_reference(x, cos, sin)

    BLOCK_HEAD = _next_power_of_two(D)
    y = torch.empty_like(x)

    _rope_fwd_kernel[(B, H, S)](
        x, cos, sin, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        S, D,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=2,  # head_dim is small; 2 warps suffice
    )
    return y


# ---------- Roofline metadata ----------


def rope_bytes(shape: tuple[int, int, int, int], dtype: torch.dtype) -> int:
    """Bytes for RoPE on a [B, H, S, D] tensor."""
    B, H, S, D = shape
    elem = torch.tensor([], dtype=dtype).element_size()
    # Read x, write y, plus cos/sin (read once, cached).
    return (2 * B * H * S * D + 2 * S * D) * elem


def rope_flops(shape: tuple[int, int, int, int]) -> int:
    B, H, S, D = shape
    # 2 mul + 1 add per element; rotation pattern doubles.
    return 6 * B * H * S * D
