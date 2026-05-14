"""Tests for the kernel name classifier."""

import pytest

from kvforge.profiler.classify import OpType, classify


@pytest.mark.parametrize("name,expected", [
    # Matmul variants
    ("Cijk_Ailk_Bljk_HHS_BH_MT128x128x16", OpType.MATMUL),  # rocBLAS-style
    ("hipblasGemmEx", OpType.MATMUL),
    ("composable_kernel_gemm_xdl", OpType.MATMUL),
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
    assert classify("HIPBLAS_GEMM") == OpType.MATMUL


def test_classify_prefers_specific_match() -> None:
    """`rms_norm` should classify as RMSNORM, not LAYERNORM, even though both
    match the more generic `layer_norm` pattern in lowercase."""
    assert classify("aten::rms_norm") == OpType.RMSNORM
