"""GEMM primitives — references and autotuned Triton templates.

Two GEMMs matter for decode:

  - **FP8 ``A @ B``** for the QKV / O / MLP projections at small M
    (M = number of decode tokens in the batch). The hipBLASLt FP8 path is
    weak on these skinny shapes — that's where slipstream wins.
  - **FP16/BF16 fall-back** for cases where FP8 quantization is too lossy
    (e.g., language-model head projection).

The reference :func:`fp8_gemm_reference` is the correctness oracle. The
Triton template lives in ``triton_kernels.py`` (added in P2).
"""

from slipstream.gemm.reference import fp16_gemm_reference, fp8_gemm_reference

__all__ = ["fp16_gemm_reference", "fp8_gemm_reference"]
