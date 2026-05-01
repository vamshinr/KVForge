"""Online (numerically stable) softmax: Triton kernel + reference.

The classic safe-softmax algorithm subtracts the row max before exponentiating.
Naive implementations require three passes over each row (max, exp+sum, div).
The "online" variant from the FlashAttention paper fuses passes 1+2 by
maintaining running (m, l) state — but for plain softmax over a single row
that fits in shared memory, a one-pass implementation is simpler and fast
enough.

This kernel uses one program per row. Rows up to BLOCK_SIZE elements are
processed entirely in registers/SMEM. Larger rows fall back to the eager
reference (still correct, just slower).

Roofline: pure memory-bound. ~2 FLOPs/byte at fp16.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard PyTorch softmax — used as the correctness oracle."""
    return torch.softmax(x, dim=dim)


if _HAS_TRITON:

    @triton.jit
    def _softmax_fwd_kernel(
        x_ptr, y_ptr,
        x_row_stride, y_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One row per program. Computes safe softmax in three fused passes."""
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        x_ptrs = x_ptr + row_idx * x_row_stride + col_offsets
        # Use -inf for masked-out elements so they don't affect the max.
        x = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

        # Pass 1: row max.
        row_max = tl.max(x, axis=0)
        # Pass 2: exp(x - max) and sum.
        x_centered = x - row_max
        numerator = tl.exp(x_centered)
        # Mask out invalid lanes from the sum.
        numerator = tl.where(mask, numerator, 0.0)
        denominator = tl.sum(numerator, axis=0)
        # Pass 3: divide.
        y = numerator / denominator

        y_ptrs = y_ptr + row_idx * y_row_stride + col_offsets
        tl.store(y_ptrs, y, mask=mask)


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Public softmax forward (last-dim only for the Triton path)."""
    if not _HAS_TRITON or not x.is_cuda or dim not in (-1, x.ndim - 1):
        return softmax_reference(x, dim=dim)

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape

    BLOCK_SIZE = _next_power_of_two(n_cols)
    if BLOCK_SIZE > 16384:
        return softmax_reference(x, dim=dim)

    y_2d = torch.empty_like(x_2d)
    _softmax_fwd_kernel[(n_rows,)](
        x_2d, y_2d,
        x_2d.stride(0), y_2d.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 if BLOCK_SIZE >= 4096 else 4,
    )
    return y_2d.reshape(orig_shape)


# ---------- Roofline metadata ----------


def softmax_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    elem = torch.tensor([], dtype=dtype).element_size()
    n = 1
    for d in shape:
        n *= d
    return 2 * n * elem  # read + write


def softmax_flops(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    # max + sub + exp + sum + div ≈ 5 FLOPs/elem
    return 5 * n
