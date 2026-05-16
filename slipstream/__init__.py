"""slipstream: a decode-optimized LLM inference path for AMD MI300X.

Paged FP8 KV cache + continuous batching + autotuned Triton kernels.
See ``docs/PLAN.md`` for the project plan.
"""

__version__ = "0.1.0"
__author__ = "Vamshi Nagireddy"

from slipstream.hardware import GPUSpec, detect_gpu, has_triton, is_rocm
from slipstream.kvcache.paged import PagedKVCache, SequenceState
from slipstream.roofline import RooflineCalculator
from slipstream.testing.harness import CorrectnessHarness

__all__ = [
    "CorrectnessHarness",
    "GPUSpec",
    "PagedKVCache",
    "RooflineCalculator",
    "SequenceState",
    "detect_gpu",
    "has_triton",
    "is_rocm",
]
