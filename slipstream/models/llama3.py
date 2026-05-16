"""Llama-3 forward path on slipstream primitives.

This module owns the **shape and weights** of the model, but defers all
performance-sensitive work to slipstream's primitives:

  - Attention: :func:`slipstream.attention.paged_attention_reference`
    (swapped for the Triton kernel in Phase 2)
  - GEMM: :func:`slipstream.gemm.fp16_gemm_reference` and
    :func:`slipstream.gemm.fp8_gemm_reference` (swapped for Triton in Phase 3)
  - Norms: :func:`slipstream.kernels.rmsnorm`
  - Rotary: :func:`slipstream.kernels.rope`

Weight quantization (FP8) is done once at load time and cached on disk.
At inference, weights are read as FP8 and decoded inline by the GEMM kernel.

Llama-3-8B reference configuration::

    n_layers          32
    hidden            4096
    intermediate      14336
    n_q_heads         32
    n_kv_heads        8       (GQA group size 4)
    head_dim          128
    vocab_size        128_256
    rope_theta        500_000
    max_position      8192
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Llama3Config:
    """The architectural constants of a Llama-3 model variant."""

    n_layers: int
    hidden: int
    intermediate: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 8192

    @classmethod
    def llama3_8b(cls) -> "Llama3Config":
        return cls(
            n_layers=32, hidden=4096, intermediate=14336,
            n_q_heads=32, n_kv_heads=8, head_dim=128,
            vocab_size=128_256,
        )

    @classmethod
    def llama3_70b(cls) -> "Llama3Config":
        return cls(
            n_layers=80, hidden=8192, intermediate=28672,
            n_q_heads=64, n_kv_heads=8, head_dim=128,
            vocab_size=128_256,
        )


# The Llama-3 forward module — a torch.nn.Module that uses the slipstream
# primitives for every hot op — lands as part of Phase 3. The skeleton here
# documents the structure so the rest of the codebase has a stable interface
# to reference.

def build_llama3(
    config: Llama3Config,
    *,
    dtype: str = "fp16",
    fp8_weights: bool = False,
    device: str = "cuda",
):
    """Return a callable that takes (packed_q, packed_kv, block_tables, seq_lens)
    and returns logits.

    Today this raises :class:`NotImplementedError`; the engine's bench adapter
    catches that and reports the baseline as "skipped, not failed."
    """
    raise NotImplementedError(
        "Llama-3 forward path lands in Phase 3 alongside the Triton kernels. "
        f"Requested config: {config}, dtype={dtype}, fp8_weights={fp8_weights}, "
        f"device={device}"
    )
