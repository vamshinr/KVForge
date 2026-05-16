"""RMSNorm: optimized Triton kernel + eager reference.

Math:
    y = x * weight / sqrt(mean(x^2) + eps)

This kernel processes one row of `x` per Triton program. The row is loaded
once into shared memory (via the L1/SMEM hierarchy that Triton manages),
the variance is computed with a single pass, and the normalized output is
written back fused with the weight multiply. Compared to the eager
PyTorch implementation — which materializes intermediate tensors for
`x.pow(2)`, `mean`, and the multiply-add chain — this saves three full
HBM round-trips on the input row.

Roofline characterization:
  - Bytes per row: 2 * N * sizeof(dtype) (read x, write y; weight is reused)
  - FLOPs per row: ~3 * N (square, accumulate, multiply)
  - Arithmetic intensity: ~1.5 FLOP/byte → memory-bound on every modern GPU.

Optimization tier (per AutoKernel playbook): Tier 1 (block sizes) + Tier 2
(memory access). Tier 3 (compute) does not apply since we're memory-bound.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------- Eager reference (correctness oracle) ----------


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Eager FP32-accumulated RMSNorm. Matches the model's nn.Module forward.

    Always cast to fp32 internally — the Triton kernel does the same — so
    correctness comparisons aren't dominated by accumulation noise.
    """
    in_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (x32 * weight).to(in_dtype)


# ---------- Triton kernel ----------


if _HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,        # input  [n_rows, n_cols]
        weight_ptr,   # weight [n_cols]
        y_ptr,        # output [n_rows, n_cols]
        x_row_stride,
        y_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row. Reads the row, computes inverse-RMS, writes back.

        BLOCK_SIZE must be >= n_cols and a power of two. For TinyLlama (cols=2048)
        BLOCK_SIZE=2048 is the natural choice; for Llama-2 7B (cols=4096) use 4096.
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load row.
        x_ptrs = x_ptr + row_idx * x_row_stride + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Single-pass variance (no separate mean-subtract: RMS not stddev).
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)

        # Apply weight and write back. Weight is broadcast — loaded once per row,
        # which the L2 cache handles efficiently across concurrent programs.
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        y_ptrs = y_ptr + row_idx * y_row_stride + col_offsets
        tl.store(y_ptrs, y, mask=mask)


def _next_power_of_two(n: int) -> int:
    """Smallest power-of-two >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Public RMSNorm forward.

    Falls back to the eager reference if Triton is unavailable or the input
    is on CPU. The 2D reshape lets us handle the standard
    ``[batch, seq_len, hidden]`` layout without per-call overhead.

    Parameters
    ----------
    x: shape ``[..., hidden]`` — any number of leading dims is fine.
    weight: shape ``[hidden]``.
    eps: numerical stability term.
    """
    if not _HAS_TRITON or not x.is_cuda:
        return rmsnorm_reference(x, weight, eps)

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x_2d.shape

    # Triton requires power-of-two BLOCK_SIZE for many reduction primitives.
    BLOCK_SIZE = _next_power_of_two(n_cols)
    if BLOCK_SIZE > 16384:
        # Past 16k cols, single-block-per-row breaks down. Fall back.
        return rmsnorm_reference(x, weight, eps)

    y_2d = torch.empty_like(x_2d)
    _rmsnorm_fwd_kernel[(n_rows,)](
        x_2d, weight, y_2d,
        x_2d.stride(0), y_2d.stride(0),
        n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 if BLOCK_SIZE >= 2048 else 4,
    )
    return y_2d.reshape(orig_shape)


# ---------- Roofline metadata ----------


def rmsnorm_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Bytes moved through DRAM for an RMSNorm with the given input shape."""
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    # Read x, write y, plus negligible weight (reused).
    return 2 * n_elements * elem_bytes


def rmsnorm_flops(shape: tuple[int, ...]) -> int:
    """Approximate FLOPs for an RMSNorm with the given input shape."""
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    # square + accumulate + sqrt + divide + multiply ≈ 5 FLOPs/elem
    return 5 * n_elements
