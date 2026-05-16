"""Autotune cache for parameterized Triton kernels.

The autotune cache is the scaling mechanism slipstream uses **instead of**
hand-tuning kernels per shape. Each parameterized template (one for paged
attention, one for FP8 GEMM, etc.) exposes its Triton config space; the
autotuner sweeps it across the decode-shape distribution and persists the
winning config to a JSON cache. At inference time, kernels look up their
config from the cache instead of recomputing.

This is conceptually similar to TorchInductor's autotune cache, but tailored
to decode-step shapes (small M, large N=K=hidden, ctx ∈ powers of 2) and
keyed by the GPU arch so the cache transports across boxes with the same
hardware. Cache misses on a new shape trigger an on-demand sweep with a
warning; production deployments are expected to run the full
``slipstream-autotune`` sweep ahead of time.

Key files:
  - :mod:`cache` — persistent JSON store, lookup/record API
  - :mod:`shape_buckets` — quantizes shapes into the discrete grid
  - :mod:`sweep` — runs a kernel across configs and records the winner
"""

from slipstream.autotune.cache import AutotuneCache, CacheKey, TunedConfig
from slipstream.autotune.shape_buckets import (
    bucket_attention_shape,
    bucket_gemm_shape,
)

__all__ = [
    "AutotuneCache",
    "CacheKey",
    "TunedConfig",
    "bucket_attention_shape",
    "bucket_gemm_shape",
]
