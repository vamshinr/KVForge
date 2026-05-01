"""Roofline performance modeling.

The roofline model classifies a kernel as compute-bound or memory-bound based
on its arithmetic intensity (FLOPs / byte) compared to the hardware's compute-
to-bandwidth ratio (the "ridge point"). This determines which optimization
tier the search loop should pursue:

  - Memory-bound  → focus on reducing HBM traffic (fusion, vectorized loads).
  - Compute-bound → focus on tile sizes, tensor core utilization, math precision.

A kernel achieves "good" performance when its measured throughput is close
to the relevant roof:

  achievable_gflops = min(peak_compute, peak_bw * arithmetic_intensity)
"""

from __future__ import annotations

from dataclasses import dataclass

from kvforge.hardware import GPUSpec


@dataclass
class RooflineResult:
    """Roofline analysis for a single measured kernel run."""

    measured_gflops: float
    measured_bw_gb_s: float
    arithmetic_intensity: float        # FLOPs / byte
    bound: str                         # 'compute' or 'memory'
    achievable_gflops: float           # roof at this AI
    pct_of_peak: float                 # measured / achievable, 0..1

    def __str__(self) -> str:
        return (
            f"AI={self.arithmetic_intensity:.2f} FLOP/B  "
            f"{self.bound}-bound  "
            f"{self.measured_gflops:.1f}/{self.achievable_gflops:.1f} GFLOPS "
            f"({self.pct_of_peak * 100:.0f}%)"
        )


class RooflineCalculator:
    """Computes roofline metrics given a GPU spec."""

    def __init__(self, gpu: GPUSpec, dtype_is_fp16: bool = True) -> None:
        self.gpu = gpu
        # Pick the appropriate compute peak for the dtype being benchmarked.
        self.peak_gflops = (
            gpu.peak_fp16_tflops if dtype_is_fp16 else gpu.peak_fp32_tflops
        ) * 1000.0
        self.peak_bw_gb_s = gpu.peak_bw_gb_s
        # Ridge point: arithmetic intensity at which compute and memory roofs cross.
        self.ridge_point = self.peak_gflops / self.peak_bw_gb_s if self.peak_bw_gb_s > 0 else 0.0

    def analyze(
        self, flops: int, bytes_moved: int, runtime_seconds: float
    ) -> RooflineResult:
        """Compute roofline metrics for a single measurement.

        Parameters
        ----------
        flops: total floating-point operations performed.
        bytes_moved: bytes read+written from DRAM.
        runtime_seconds: measured wall-clock latency.
        """
        if runtime_seconds <= 0:
            raise ValueError("runtime_seconds must be > 0")

        measured_gflops = flops / runtime_seconds / 1e9
        measured_bw_gb_s = bytes_moved / runtime_seconds / 1e9
        ai = flops / bytes_moved if bytes_moved > 0 else float("inf")

        # Achievable (theoretical) throughput at this AI.
        achievable_gflops = min(self.peak_gflops, self.peak_bw_gb_s * ai)
        bound = "memory" if ai < self.ridge_point else "compute"
        pct = measured_gflops / achievable_gflops if achievable_gflops > 0 else 0.0

        return RooflineResult(
            measured_gflops=measured_gflops,
            measured_bw_gb_s=measured_bw_gb_s,
            arithmetic_intensity=ai,
            bound=bound,
            achievable_gflops=achievable_gflops,
            pct_of_peak=min(pct, 1.0),
        )

    def recommend_tier(self, result: RooflineResult) -> list[str]:
        """Suggest optimization tiers given a roofline result.

        These are textual hints — in a real agent loop they'd seed the
        candidate generator's prompts. In this static framework they show up
        in benchmark reports and CLI output.
        """
        tiers: list[str] = []
        if result.pct_of_peak >= 0.85:
            return ["near-peak: investigate kernel-specific algorithmic improvements"]

        if result.bound == "memory":
            tiers.append("Tier 2 — coalesced loads, vectorized memory access (float4/half2)")
            tiers.append("Tier 1 — sweep BLOCK_SIZE, num_warps, num_stages")
            if result.pct_of_peak < 0.5:
                tiers.append("Tier 3 — fuse with adjacent ops to amortize HBM round-trips")
        else:  # compute-bound
            tiers.append("Tier 1 — tile dim sweep with focus on 128×128 / 256×128 variants")
            tiers.append("Tier 4 — split-K, persistent kernels, tensor-core instruction selection")
            if result.pct_of_peak < 0.5:
                tiers.append("Tier 3 — accumulator precision (TF32 / FP16+FP32 mixed)")

        return tiers
