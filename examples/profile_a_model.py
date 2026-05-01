"""Example: profile a model and inspect the kernel mix.

Usage:
    python examples/profile_a_model.py
"""

from __future__ import annotations

import torch

from kvforge.models.tinyllama import build_tinyllama, make_forward_fn
from kvforge.profiler import ModelProfiler


def main() -> None:
    model = build_tinyllama(tiny=True)  # Tiny so the example runs on CPU
    forward_fn = make_forward_fn(batch_size=1, seq_len=64, mode="prefill")

    profiler = ModelProfiler(warmup_iters=3, measured_iters=5)
    result = profiler.profile(model, forward_fn)

    print(f"Total GPU time / iter: {result.total_gpu_us:.0f} µs")
    print(f"Number of kernels:     {len(result.entries)}")
    print(f"Number of op-types:    {len(result.aggregated)}")
    print()
    print(f"{'#':>3}  {'op':<12}  {'µs':>8}  {'%':>6}  {'2× proj':>8}  {'5× proj':>8}")
    print("-" * 60)
    for entry in result.aggregated[:10]:
        proj_2x = entry.projections.get(2.0, 1.0)
        proj_5x = entry.projections.get(5.0, 1.0)
        print(f"{entry.rank:>3}  {entry.op_type.value:<12}  "
              f"{entry.total_us:>8.1f}  {entry.fraction*100:>5.1f}%  "
              f"{proj_2x:>7.2f}×  {proj_5x:>7.2f}×")


if __name__ == "__main__":
    main()
