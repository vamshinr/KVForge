"""Iterative keep/revert search loop.

This is a simplified, non-LLM version of the AutoKernel agent loop. Instead
of having an LLM generate candidate kernels, the loop accepts an explicit
list of candidate variants (each implementing the same call signature) and
evaluates them in turn. This makes the loop testable, deterministic, and
educational without requiring API keys or a language model.

For a production agent loop, replace `CandidateGenerator` with one that calls
out to an LLM and writes a single kernel file per iteration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

from kvforge.optimizer.harness import CorrectnessHarness, KernelFn


@dataclass
class SearchEntry:
    """A single iteration's record."""

    iteration: int
    candidate_label: str
    passed_correctness: bool
    failed_stage: str | None
    runtime_ms: float | None
    decision: str   # 'keep' | 'revert' | 'rejected'
    note: str = ""


@dataclass
class SearchHistory:
    """Append-only record of an entire search run."""

    entries: list[SearchEntry] = field(default_factory=list)

    @property
    def n_kept(self) -> int:
        return sum(1 for e in self.entries if e.decision == "keep")

    @property
    def n_reverted(self) -> int:
        return sum(1 for e in self.entries if e.decision == "revert")

    def consecutive_reverts(self) -> int:
        n = 0
        for e in reversed(self.entries):
            if e.decision == "revert":
                n += 1
            else:
                break
        return n

    def best(self) -> SearchEntry | None:
        kept = [e for e in self.entries if e.decision == "keep" and e.runtime_ms]
        if not kept:
            return None
        return min(kept, key=lambda e: e.runtime_ms)  # type: ignore[arg-type]


class SearchLoop:
    """Run the keep/revert loop over candidate kernels.

    Move-on criteria match AutoKernel's defaults:
      - 5 consecutive reverts → plateau, give up
      - target speedup reached → done
      - time budget exhausted → done
    """

    def __init__(
        self,
        baseline_fn: KernelFn,
        harness: CorrectnessHarness,
        bench_input_factory: Callable[[], tuple[tuple, dict]],
        max_consecutive_reverts: int = 5,
        target_speedup: float = 2.0,
        time_budget_s: float = 600.0,
        bench_iters: int = 50,
        bench_warmup: int = 10,
        improvement_threshold: float = 1.01,
    ) -> None:
        self.baseline_fn = baseline_fn
        self.harness = harness
        self.bench_input_factory = bench_input_factory
        self.max_consecutive_reverts = max_consecutive_reverts
        self.target_speedup = target_speedup
        self.time_budget_s = time_budget_s
        self.bench_iters = bench_iters
        self.bench_warmup = bench_warmup
        self.improvement_threshold = improvement_threshold

    def run(
        self,
        candidates: Iterator[tuple[str, KernelFn]],
    ) -> SearchHistory:
        """Iterate through `candidates` until a stop criterion fires.

        Each candidate is a `(label, kernel_fn)` pair. The loop:
          1. Validates correctness via the harness.
          2. Benchmarks the candidate.
          3. Compares to the current best (or baseline if none kept yet).
          4. Records `keep` if faster by `improvement_threshold`, else `revert`.
        """
        history = SearchHistory()
        # Establish baseline runtime once.
        baseline_ms = self._bench(self.baseline_fn)
        best_ms = baseline_ms

        loop_start = time.perf_counter()
        for i, (label, candidate_fn) in enumerate(candidates, start=1):
            elapsed = time.perf_counter() - loop_start
            if elapsed > self.time_budget_s:
                history.entries.append(SearchEntry(
                    i, label, False, None, None, "rejected",
                    "time budget exhausted"))
                break
            if history.consecutive_reverts() >= self.max_consecutive_reverts:
                history.entries.append(SearchEntry(
                    i, label, False, None, None, "rejected",
                    f"plateau: {self.max_consecutive_reverts} consecutive reverts"))
                break
            if best_ms > 0 and (baseline_ms / best_ms) >= self.target_speedup:
                history.entries.append(SearchEntry(
                    i, label, False, None, None, "rejected",
                    f"target speedup {self.target_speedup}x reached"))
                break

            # Stage 1: correctness.
            corr = self.harness.validate(candidate_fn)
            if not corr.passed:
                history.entries.append(SearchEntry(
                    i, label, False, corr.failed_stage, None, "revert",
                    corr.failure_detail[:200]))
                continue

            # Stage 2: benchmark.
            cand_ms = self._bench(candidate_fn)
            decision = "keep" if cand_ms < best_ms / self.improvement_threshold else "revert"
            note = (
                f"{baseline_ms / cand_ms:.2f}x vs baseline, "
                f"{best_ms / cand_ms:.2f}x vs best"
            )
            history.entries.append(SearchEntry(
                i, label, True, None, cand_ms, decision, note))
            if decision == "keep":
                best_ms = cand_ms

        return history

    def _bench(self, fn: KernelFn) -> float:
        """Median-of-N timing in milliseconds."""
        args, kwargs = self.bench_input_factory()
        # Warmup.
        for _ in range(self.bench_warmup):
            _ = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(self.bench_iters):
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fn(*args, **kwargs)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = fn(*args, **kwargs)
                timings.append((time.perf_counter() - t0) * 1000)

        timings.sort()
        # Trimmed median (drop top/bottom 10%).
        lo = max(1, len(timings) // 10)
        trimmed = timings[lo:-lo] if len(timings) > 2 * lo else timings
        return sum(trimmed) / len(trimmed)
