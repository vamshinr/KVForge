"""Benchmark harness: compares baselines and reports roofline-aware metrics.

Three baselines per kernel:
  - eager:    pure PyTorch reference (cuBLAS / ATen).
  - compile:  `torch.compile(fn, mode='max-autotune')`.
  - kvforge:  the optimized Triton kernel.

Each is timed with CUDA events, 200 iterations, trimmed mean (drop top/bottom
10%). Results include throughput and roofline percent-of-peak.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from kvforge.hardware import GPUSpec, detect_gpu
from kvforge.optimizer.roofline import RooflineCalculator, RooflineResult


@dataclass
class BenchmarkResult:
    """Single (kernel, baseline, shape) measurement."""

    kernel: str
    baseline: str           # 'eager' | 'compile' | 'kvforge'
    shape: tuple[int, ...]
    dtype: torch.dtype
    runtime_us: float
    roofline: RooflineResult | None = None


@dataclass
class BenchmarkSuite:
    """Aggregate results for a single kernel across baselines."""

    kernel: str
    shape: tuple[int, ...]
    eager: BenchmarkResult | None = None
    compile: BenchmarkResult | None = None
    kvforge: BenchmarkResult | None = None
    extras: dict[str, BenchmarkResult] = field(default_factory=dict)

    def speedup(self, vs: str = "eager") -> float | None:
        if self.kvforge is None:
            return None
        ref = getattr(self, vs, None)
        if ref is None:
            return None
        return ref.runtime_us / self.kvforge.runtime_us if self.kvforge.runtime_us > 0 else None


class BenchmarkHarness:
    """Times a kernel function across multiple baselines."""

    def __init__(
        self,
        gpu: GPUSpec | None = None,
        warmup_iters: int = 25,
        bench_iters: int = 200,
        trim_pct: float = 0.10,
    ) -> None:
        self.gpu = gpu or detect_gpu()
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.trim_pct = trim_pct

    def time_fn(
        self,
        fn: Callable[[], torch.Tensor],
    ) -> float:
        """Return median (trimmed) runtime in microseconds."""
        # Warmup.
        for _ in range(self.warmup_iters):
            _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        timings_ms: list[float] = []
        if torch.cuda.is_available():
            for _ in range(self.bench_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fn()
                end.record()
                end.synchronize()
                timings_ms.append(start.elapsed_time(end))
        else:
            for _ in range(self.bench_iters):
                t0 = time.perf_counter()
                _ = fn()
                timings_ms.append((time.perf_counter() - t0) * 1000)

        timings_ms.sort()
        trim = max(1, int(len(timings_ms) * self.trim_pct))
        trimmed = (
            timings_ms[trim:-trim] if len(timings_ms) > 2 * trim else timings_ms
        )
        median_ms = sum(trimmed) / len(trimmed)
        return median_ms * 1000  # convert ms → µs

    def benchmark_three_way(
        self,
        kernel_name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        eager_fn: Callable[[], torch.Tensor],
        kvforge_fn: Callable[[], torch.Tensor],
        compile_fn: Callable[[], torch.Tensor] | None = None,
        flops: int | None = None,
        bytes_moved: int | None = None,
    ) -> BenchmarkSuite:
        """Benchmark a kernel under all three baselines.

        `flops` and `bytes_moved` enable roofline analysis. If omitted, only
        wall-clock times are reported.
        """
        suite = BenchmarkSuite(kernel=kernel_name, shape=shape)
        roofline = RooflineCalculator(
            self.gpu, dtype_is_fp16=(dtype in (torch.float16, torch.bfloat16))
        )

        for label, fn in [("eager", eager_fn), ("compile", compile_fn), ("kvforge", kvforge_fn)]:
            if fn is None:
                continue
            us = self.time_fn(fn)
            r: RooflineResult | None = None
            if flops is not None and bytes_moved is not None and us > 0:
                r = roofline.analyze(flops, bytes_moved, us / 1e6)
            result = BenchmarkResult(
                kernel=kernel_name, baseline=label, shape=shape,
                dtype=dtype, runtime_us=us, roofline=r,
            )
            setattr(suite, label, result)
        return suite
