"""Tests for the kernel name classifier."""

import pytest

from kvforge.profiler.classify import OpType, classify


@pytest.mark.parametrize("name,expected", [
    # cuBLAS / cuDNN matmul variants
    ("ampere_sgemm_64x64_nn", OpType.MATMUL),
    ("cublasGemmEx", OpType.MATMUL),
    ("cutlass_80_tensorop_s16816gemm", OpType.MATMUL),
    ("aten::addmm", OpType.MATMUL),
    ("aten::linear", OpType.MATMUL),
    # Attention
    ("flash_attn_fwd", OpType.ATTENTION),
    ("scaled_dot_product_attention", OpType.ATTENTION),
    ("aten::sdpa", OpType.ATTENTION),
    # Norms
    ("aten::rms_norm", OpType.RMSNORM),
    ("triton_rmsnorm_fwd_kernel", OpType.RMSNORM),
    ("aten::layer_norm", OpType.LAYERNORM),
    # Softmax
    ("aten::softmax", OpType.SOFTMAX),
    ("aten::log_softmax", OpType.SOFTMAX),
    # RoPE
    ("apply_rotary_emb", OpType.ROPE),
    # Activations
    ("aten::silu", OpType.SILU),
    ("aten::gelu", OpType.GELU),
    # Embedding
    ("aten::embedding", OpType.EMBEDDING),
    ("aten::index_select", OpType.EMBEDDING),
    # Should NOT match anything specific
    ("some_unknown_kernel_xyz123", OpType.OTHER),
    ("", OpType.OTHER),
])
def test_classify_known_kernels(name: str, expected: OpType) -> None:
    assert classify(name) == expected


def test_classify_is_case_insensitive() -> None:
    assert classify("FLASH_ATTN_FWD") == OpType.ATTENTION
    assert classify("CUBLAS_GEMM") == OpType.MATMUL


def test_classify_prefers_specific_match() -> None:
    """`rms_norm` should classify as RMSNORM, not LAYERNORM, even though both
    match the more generic `layer_norm` pattern in lowercase."""
    assert classify("aten::rms_norm") == OpType.RMSNORM
