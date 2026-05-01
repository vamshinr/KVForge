"""Example: validate and benchmark an optimized kernel against its reference.

Demonstrates the correctness harness + search loop on RMSNorm.

Usage:
    python examples/optimize_a_kernel.py
"""

from __future__ import annotations

import torch

from kvforge.kernels.rmsnorm import rmsnorm, rmsnorm_reference
from kvforge.optimizer.harness import CorrectnessHarness
from kvforge.optimizer.search import SearchLoop


def make_inputs(shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device)
    return (x, w), {"eps": 1e-5}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    harness = CorrectnessHarness(
        reference_fn=rmsnorm_reference,
        input_factory=make_inputs,
        shape_configs=[(4, 2048), (16, 4096), (1, 2048)],
        edge_shapes=[(7, 2047), (5, 1023)],
        device=device,
    )

    # Validate the optimized kernel passes all five stages.
    print("Validating optimized RMSNorm against reference...")
    result = harness.validate(rmsnorm)
    if not result.passed:
        print(f"FAILED at stage '{result.failed_stage}': {result.failure_detail}")
        return
    print("All correctness stages passed.")
    for stage, ms in result.stage_times_ms.items():
        print(f"  {stage:<14}: {ms:>6.1f} ms")

    # Now benchmark.
    bench_shape = (16, 4096)
    bench_dtype = torch.float16 if device.type == "cuda" else torch.float32

    def bench_factory():
        return make_inputs(bench_shape, bench_dtype, device)

    loop = SearchLoop(
        baseline_fn=rmsnorm_reference,
        harness=harness,
        bench_input_factory=bench_factory,
        bench_iters=50,
        bench_warmup=10,
    )

    baseline_ms = loop._bench(rmsnorm_reference)
    optimized_ms = loop._bench(rmsnorm)
    print()
    print(f"RMSNorm @ shape={bench_shape}, dtype={bench_dtype}:")
    print(f"  reference: {baseline_ms:.3f} ms")
    print(f"  optimized: {optimized_ms:.3f} ms")
    print(f"  speedup:   {baseline_ms / optimized_ms:.2f}×")


if __name__ == "__main__":
    main()
