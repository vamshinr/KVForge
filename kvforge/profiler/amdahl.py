"""Amdahl's-law-based kernel ranking.

Given a per-kernel time table from the profiler, ranks kernels by the
end-to-end speedup they would deliver under hypothetical local speedups.
This drives where the optimization loop spends its budget.

Amdahl's law:

    S_total = 1 / ((1 - f) + f / s)

where `f` is the fraction of total runtime spent in the kernel and `s` is the
local speedup achieved on that kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kvforge.profiler.classify import OpType


@dataclass
class KernelEntry:
    """A single kernel's profile entry plus what-if projections."""

    name: str
    op_type: OpType
    total_us: float          # cumulative time across all calls
    call_count: int
    fraction: float = 0.0    # fraction of total wall-clock GPU time
    rank: int = 0            # 1 = highest impact

    # What-if speedups: end-to-end S_total for given local speedup s.
    projections: dict[float, float] = field(default_factory=dict)

    def project(self, local_speedup: float) -> float:
        """Compute end-to-end speedup if this kernel sped up by `local_speedup`x.

        Returns S_total such that 1.0 means no change. Always >= 1.0 for any
        local_speedup >= 1.0.
        """
        if local_speedup <= 0:
            raise ValueError("local_speedup must be positive")
        f = self.fraction
        return 1.0 / ((1.0 - f) + f / local_speedup)


class AmdahlRanker:
    """Ranks profiled kernels by their Amdahl impact on end-to-end latency."""

    # Local speedups to project for each kernel. 1.5x and 2x are realistic for
    # memory-bound kernels with good optimization; 3x and 5x are aspirational.
    DEFAULT_PROJECTIONS = (1.5, 2.0, 3.0, 5.0)

    def __init__(self, projections: tuple[float, ...] = DEFAULT_PROJECTIONS) -> None:
        self.projections = projections

    def rank(
        self,
        kernel_times: dict[str, tuple[float, int]],
        op_types: dict[str, OpType] | None = None,
    ) -> list[KernelEntry]:
        """Build a ranked list of `KernelEntry` objects.

        Parameters
        ----------
        kernel_times: mapping of kernel name -> (total_microseconds, call_count).
        op_types: optional pre-classified op types per kernel; if omitted, the
            ranker leaves `OpType.OTHER` and the caller is expected to classify.

        Returns
        -------
        A list of `KernelEntry`, sorted by descending fraction of total time.
        Each entry's `projections` dict is populated for the ranker's
        configured local speedups.
        """
        op_types = op_types or {}
        total = sum(t for t, _ in kernel_times.values())
        if total <= 0:
            return []

        entries: list[KernelEntry] = []
        for name, (total_us, call_count) in kernel_times.items():
            entry = KernelEntry(
                name=name,
                op_type=op_types.get(name, OpType.OTHER),
                total_us=total_us,
                call_count=call_count,
                fraction=total_us / total,
            )
            entry.projections = {s: entry.project(s) for s in self.projections}
            entries.append(entry)

        entries.sort(key=lambda e: e.fraction, reverse=True)
        for i, e in enumerate(entries, start=1):
            e.rank = i
        return entries

    def aggregate_by_op_type(
        self, entries: list[KernelEntry]
    ) -> list[KernelEntry]:
        """Collapse multiple kernel entries that share an op_type.

        Useful when many cuBLAS GEMM variants all map to MATMUL — the Amdahl
        impact of optimizing "matmul as a class" is larger than any single
        variant.
        """
        bucket: dict[OpType, KernelEntry] = {}
        for e in entries:
            if e.op_type not in bucket:
                bucket[e.op_type] = KernelEntry(
                    name=f"<all {e.op_type.value}>",
                    op_type=e.op_type,
                    total_us=0.0,
                    call_count=0,
                )
            agg = bucket[e.op_type]
            agg.total_us += e.total_us
            agg.call_count += e.call_count

        total = sum(b.total_us for b in bucket.values())
        if total <= 0:
            return []
        out = list(bucket.values())
        for b in out:
            b.fraction = b.total_us / total
            b.projections = {s: b.project(s) for s in self.projections}
        out.sort(key=lambda e: e.fraction, reverse=True)
        for i, e in enumerate(out, start=1):
            e.rank = i
        return out
