"""End-to-end decode engine.

The :class:`Engine` wires the four foundation pieces together:

  - :class:`slipstream.kvcache.PagedKVCache` — global KV pool
  - :class:`slipstream.scheduler.Scheduler` — continuous batcher
  - :mod:`slipstream.attention` kernels — decode attention
  - :mod:`slipstream.gemm` kernels — projections / MLPs

It loads HuggingFace weights, optionally quantizes them to FP8, and drives
the scheduler's step loop until all in-flight requests are done.

Before the Triton kernels land in Phase 2 this class is runnable but slow
(reference impls only) — that's intentional. Correctness first, perf later.
The bench adapter (:mod:`slipstream.bench.baselines.slipstream_adapter`)
talks to this class.
"""

from slipstream.engine.engine import Engine, EngineConfig

__all__ = ["Engine", "EngineConfig"]
