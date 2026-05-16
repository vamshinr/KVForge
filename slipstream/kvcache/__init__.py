"""Paged KV cache with optional FP8 quantization.

The cache is the system-level lever in slipstream. Two layers:

  - ``fp8`` — pure quantize/dequantize math. Per-token-per-head scaling.
  - ``paged`` — block-table allocator, sequence state, append/read API.

The paged cache stores K and V in fixed-size blocks. A ``BlockTable`` maps a
sequence's logical token positions to physical block ids, so blocks are never
moved when a sequence grows — only appended. Block size is **16** by default
to match vLLM's convention, which makes correctness diffing block-by-block
against a vLLM reference straightforward.

The attention kernels read directly from this layout. The reference impl in
``slipstream.attention.reference`` is the ground truth that any Triton
variant must match.
"""

from slipstream.kvcache.fp8 import (
    FP8_E4M3_MAX,
    dequantize_fp8,
    quantize_fp8,
)
from slipstream.kvcache.paged import (
    BlockTable,
    PagedKVCache,
    SequenceState,
)

__all__ = [
    "BlockTable",
    "FP8_E4M3_MAX",
    "PagedKVCache",
    "SequenceState",
    "dequantize_fp8",
    "quantize_fp8",
]
