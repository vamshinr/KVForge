"""Optimized Triton kernels for inference workloads.

Each kernel ships with:
  - a Triton implementation
  - an eager PyTorch reference (correctness oracle)
  - dtype-specific tolerances
  - FLOPS / bytes formulas for roofline analysis

If Triton is unavailable, the public API falls back to the reference
implementation so unit tests still run on CPU-only machines.
"""

from kvforge.kernels.rmsnorm import rmsnorm, rmsnorm_reference
from kvforge.kernels.rope import rope, rope_reference
from kvforge.kernels.softmax import softmax, softmax_reference

__all__ = [
    "rmsnorm",
    "rmsnorm_reference",
    "rope",
    "rope_reference",
    "softmax",
    "softmax_reference",
]
