"""End-to-end model profiler.

Runs a real forward pass under `torch.profiler` and aggregates per-kernel CUDA
times. Designed to be deterministic across runs (fixed warmup, fixed iteration
count, trimmed mean) so the Amdahl ranking is stable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from kvforge.profiler.amdahl import AmdahlRanker, KernelEntry
from kvforge.profiler.classify import OpType, classify


@dataclass
class ProfileResult:
    """Output of a profiling run.

    Attributes
    ----------
    entries: per-kernel ranked entries.
    aggregated: same data collapsed by `OpType`.
    total_gpu_us: cumulative GPU time across all kernels (sanity check).
    warmup_iters / measured_iters: profiling configuration used.
    """

    entries: list[KernelEntry]
    aggregated: list[KernelEntry]
    total_gpu_us: float
    warmup_iters: int
    measured_iters: int
    raw_kernel_times: dict[str, tuple[float, int]] = field(default_factory=dict)

    def top_n(self, n: int = 10, aggregated: bool = True) -> list[KernelEntry]:
        """Return the top-n kernels by Amdahl impact."""
        return (self.aggregated if aggregated else self.entries)[:n]


class ModelProfiler:
    """Profiles a PyTorch model and ranks its kernels by Amdahl impact.

    Parameters
    ----------
    warmup_iters: forward passes before profiling starts (kernel JIT, autotuning).
    measured_iters: forward passes whose timings are aggregated.
    """

    def __init__(self, warmup_iters: int = 5, measured_iters: int = 10) -> None:
        if warmup_iters < 1 or measured_iters < 1:
            raise ValueError("iteration counts must be >= 1")
        self.warmup_iters = warmup_iters
        self.measured_iters = measured_iters
        self.ranker = AmdahlRanker()

    def profile(
        self,
        model: torch.nn.Module,
        forward_fn: Callable[[torch.nn.Module], torch.Tensor],
    ) -> ProfileResult:
        """Profile `model` by repeatedly calling `forward_fn(model)`.

        `forward_fn` is a callable so the user controls input shapes, KV cache
        state, and any decoding loop. This keeps the profiler oblivious to
        whether we're measuring prefill, decode, or full-generation latency.
        """
        model.eval()
        # Warmup outside the profiler: triggers Triton/Inductor JIT, allocator
        # settling, and any one-time cuBLAS handle creation.
        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                forward_fn(model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        with torch.inference_mode(), profile(
            activities=activities,
            record_shapes=False,  # shapes are recorded per-kernel by us, not the profiler
            with_stack=False,
            with_flops=False,
        ) as prof:
            for _ in range(self.measured_iters):
                forward_fn(model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return self._aggregate(prof)

    def _aggregate(self, prof: profile) -> ProfileResult:
        """Convert raw profiler events into an Amdahl-ranked result."""
        kernel_times: dict[str, list[float]] = defaultdict(list)
        kernel_counts: dict[str, int] = defaultdict(int)

        # `key_averages()` gives one row per unique kernel name with self CUDA
        # time already aggregated across calls. We then divide by measured_iters
        # to get per-iteration times.
        for evt in prof.key_averages():
            # `cuda_time_total` is the post-2.1 attribute; older versions used
            # `cuda_time`. Handle both gracefully.
            cuda_us = (
                getattr(evt, "self_cuda_time_total", None)
                or getattr(evt, "cuda_time_total", None)
                or 0
            )
            if cuda_us == 0:
                continue
            name = evt.key
            kernel_times[name].append(float(cuda_us))
            kernel_counts[name] += int(evt.count)

        # Per-iteration total and count.
        per_kernel: dict[str, tuple[float, int]] = {}
        for name, times in kernel_times.items():
            per_iter_us = sum(times) / max(self.measured_iters, 1)
            per_iter_count = max(kernel_counts[name] // max(self.measured_iters, 1), 1)
            per_kernel[name] = (per_iter_us, per_iter_count)

        op_types = {name: classify(name) for name in per_kernel}
        entries = self.ranker.rank(per_kernel, op_types)
        aggregated = self.ranker.aggregate_by_op_type(entries)
        total_us = sum(t for t, _ in per_kernel.values())

        return ProfileResult(
            entries=entries,
            aggregated=aggregated,
            total_gpu_us=total_us,
            warmup_iters=self.warmup_iters,
            measured_iters=self.measured_iters,
            raw_kernel_times=per_kernel,
        )
