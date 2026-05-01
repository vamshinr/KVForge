"""Kernel name classification.

Maps the noisy kernel names emitted by `torch.profiler` to a small set of
canonical operation types. Vendor names vary widely (cuBLAS GEMM has dozens
of variants, ATen kernels are templated, Triton-compiled kernels embed hash
suffixes), so the classifier uses ordered substring matching with the most
specific patterns first.
"""

from __future__ import annotations

import re
from enum import Enum


class OpType(str, Enum):
    """Canonical kernel operation types."""

    MATMUL = "matmul"
    ATTENTION = "attention"
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"
    SOFTMAX = "softmax"
    ROPE = "rope"
    SILU = "silu"
    GELU = "gelu"
    EMBEDDING = "embedding"
    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    COPY = "copy"
    OTHER = "other"


# Order matters: patterns are tested top-to-bottom, first match wins.
_PATTERNS: list[tuple[OpType, list[re.Pattern[str]]]] = [
    (OpType.ATTENTION, [
        re.compile(r"flash[_-]?attn", re.I),
        re.compile(r"scaled_dot_product", re.I),
        re.compile(r"sdpa", re.I),
        re.compile(r"paged[_-]?attn", re.I),
    ]),
    (OpType.RMSNORM, [
        re.compile(r"rms[_-]?norm", re.I),
    ]),
    (OpType.LAYERNORM, [
        re.compile(r"layer[_-]?norm", re.I),
        re.compile(r"\bln\b", re.I),
    ]),
    (OpType.SOFTMAX, [
        re.compile(r"softmax", re.I),
        re.compile(r"log_softmax", re.I),
    ]),
    (OpType.ROPE, [
        re.compile(r"rotary", re.I),
        re.compile(r"\brope\b", re.I),
    ]),
    (OpType.SILU, [
        re.compile(r"silu", re.I),
        re.compile(r"swish", re.I),
        re.compile(r"swiglu", re.I),
    ]),
    (OpType.GELU, [
        re.compile(r"gelu", re.I),
    ]),
    (OpType.MATMUL, [
        re.compile(r"gemm", re.I),         # catches sgemm, hgemm, dgemm, gemm_ex, etc.
        re.compile(r"matmul", re.I),
        re.compile(r"\bmm\b", re.I),
        re.compile(r"\bbmm\b", re.I),
        re.compile(r"linear", re.I),
        re.compile(r"cublas", re.I),
        re.compile(r"cutlass", re.I),
        re.compile(r"addmm", re.I),
    ]),
    (OpType.EMBEDDING, [
        re.compile(r"embedding", re.I),
        re.compile(r"index_select", re.I),
    ]),
    (OpType.REDUCE, [
        re.compile(r"reduce", re.I),
        re.compile(r"\bsum\b", re.I),
        re.compile(r"\bmean\b", re.I),
    ]),
    (OpType.COPY, [
        re.compile(r"copy", re.I),
        re.compile(r"memcpy", re.I),
        re.compile(r"contiguous", re.I),
    ]),
    (OpType.ELEMENTWISE, [
        re.compile(r"add", re.I),
        re.compile(r"mul", re.I),
        re.compile(r"div", re.I),
        re.compile(r"sub", re.I),
        re.compile(r"sqrt", re.I),
        re.compile(r"\bexp\b", re.I),
        re.compile(r"\blog\b", re.I),
    ]),
]


def classify(kernel_name: str) -> OpType:
    """Map a profiler kernel name to a canonical OpType.

    The matching is intentionally conservative: when in doubt, return OTHER.
    Misclassification is worse than no classification because it pollutes the
    Amdahl ranking.

    Parameters
    ----------
    kernel_name: raw kernel name string from `torch.profiler`.

    Returns
    -------
    The most specific matching `OpType`, or `OpType.OTHER` if no pattern matches.
    """
    if not kernel_name:
        return OpType.OTHER
    for op_type, patterns in _PATTERNS:
        if any(p.search(kernel_name) for p in patterns):
            return op_type
    return OpType.OTHER
