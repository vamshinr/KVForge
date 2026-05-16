"""Model implementations that use slipstream's kernels.

Currently:

  - :mod:`slipstream.models.llama3` — Llama-3 forward path using paged KV
    cache, GQA, RoPE, RMSNorm. Loads HF weights, optionally quantizes
    linear weights to FP8 on load.
"""

from slipstream.models.llama3 import Llama3Config, build_llama3

__all__ = ["Llama3Config", "build_llama3"]
