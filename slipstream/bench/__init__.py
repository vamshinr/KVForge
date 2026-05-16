"""Benchmark harness.

Two layers:

  - **Kernel benchmarks** (:mod:`slipstream.bench.harness`) — measure a single
    kernel function across eager / compile / slipstream variants. Used for
    iterating on a single Triton template.
  - **Engine benchmarks** (:mod:`slipstream.bench.runner`) — run a full
    decode workload through one or more baselines (slipstream, vLLM-ROCm,
    Flash-Attention-ROCm, hipBLASLt, torch.compile) and compare.

The CLI (`slipstream-bench`) drives the engine benchmarks; the kernel harness
is used directly from Python during kernel development.
"""

from slipstream.bench.baselines import (
    BaselineProtocol,
    BenchmarkInput,
    DecodeResult,
    list_baselines,
    make_baseline,
)
from slipstream.bench.harness import (
    BenchmarkHarness,
    BenchmarkResult,
    BenchmarkSuite,
)
from slipstream.bench.runner import run_grid, run_one
from slipstream.bench.workloads import production_decode_grid, smoke_grid

__all__ = [
    "BaselineProtocol",
    "BenchmarkHarness",
    "BenchmarkInput",
    "BenchmarkResult",
    "BenchmarkSuite",
    "DecodeResult",
    "list_baselines",
    "make_baseline",
    "production_decode_grid",
    "run_grid",
    "run_one",
    "smoke_grid",
]
