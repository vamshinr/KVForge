"""CLI: `python -m kvforge.optimize`."""

from __future__ import annotations

import argparse
import sys

import torch
from rich.console import Console
from rich.table import Table

from kvforge.hardware import detect_gpu, has_triton
from kvforge.kernels import rmsnorm, rmsnorm_reference, rope, rope_reference, softmax, softmax_reference
from kvforge.optimizer.harness import CorrectnessHarness
from kvforge.optimizer.roofline import RooflineCalculator
from kvforge.optimizer.search import SearchLoop


def _rmsnorm_inputs(shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device)
    return (x, w), {"eps": 1e-5}


def _rope_inputs(shape, dtype, device):
    # Coerce 4D shape: [B, H, S, D]
    if len(shape) == 2:
        shape = (1, 4, shape[0], shape[1])
    x = torch.randn(*shape, dtype=dtype, device=device)
    s, d = shape[2], shape[3]
    angles = torch.arange(s, device=device).float()[:, None] * torch.arange(0, d, 2, device=device).float()
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)
    return (x, cos, sin), {}


def _softmax_inputs(shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    return (x,), {"dim": -1}


KERNEL_REGISTRY = {
    "rmsnorm": {
        "candidate": rmsnorm,
        "reference": rmsnorm_reference,
        "input_factory": _rmsnorm_inputs,
        "shapes": [(4, 2048), (16, 2048), (32, 4096)],
        "edge_shapes": [(7, 2047), (5, 1023)],
    },
    "rope": {
        "candidate": rope,
        "reference": rope_reference,
        "input_factory": _rope_inputs,
        "shapes": [(1, 4, 512, 64), (1, 8, 1024, 128), (2, 4, 256, 64)],
        "edge_shapes": [(1, 4, 1023, 64)],
    },
    "softmax": {
        "candidate": softmax,
        "reference": softmax_reference,
        "input_factory": _softmax_inputs,
        "shapes": [(4, 2048), (16, 4096), (32, 1024)],
        "edge_shapes": [(7, 2047), (5, 1023)],
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvforge-optimize")
    parser.add_argument("--kernels", default="rmsnorm,rope,softmax",
                        help="comma-separated kernel names from the registry")
    parser.add_argument("--budget", type=int, default=20,
                        help="(unused in this static demo; reserved for agent loop)")
    args = parser.parse_args(argv)

    console = Console()
    gpu = detect_gpu()
    console.print(f"[bold cyan]Hardware:[/bold cyan] {gpu.name}")
    console.print(f"[bold cyan]Triton available:[/bold cyan] {has_triton()}")

    if not has_triton():
        console.print("[yellow]Triton unavailable — running correctness checks only "
                      "(speedup will be 1.0x). Install with `pip install triton` on a "
                      "CUDA-capable host to benchmark the optimized kernels.[/yellow]")

    table = Table(title="KVForge optimization summary")
    table.add_column("Kernel")
    table.add_column("Correctness", justify="center")
    table.add_column("Reference (ms)", justify="right")
    table.add_column("Optimized (ms)", justify="right")
    table.add_column("Speedup", justify="right")

    for kname in args.kernels.split(","):
        kname = kname.strip()
        if kname not in KERNEL_REGISTRY:
            console.print(f"[red]unknown kernel: {kname}[/red]")
            continue
        spec = KERNEL_REGISTRY[kname]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        harness = CorrectnessHarness(
            reference_fn=spec["reference"],
            input_factory=spec["input_factory"],
            shape_configs=spec["shapes"],
            edge_shapes=spec["edge_shapes"],
            device=device,
        )

        corr = harness.validate(spec["candidate"])
        if not corr.passed:
            table.add_row(kname, "FAIL",
                          "-", "-", f"[red]{corr.failed_stage}: {corr.failure_detail[:50]}[/red]")
            continue

        # Benchmark on the largest configured shape.
        shape = spec["shapes"][-1]
        bench_factory = lambda: spec["input_factory"](shape, torch.float16 if device.type == "cuda" else torch.float32, device)

        loop = SearchLoop(
            baseline_fn=spec["reference"],
            harness=harness,
            bench_input_factory=bench_factory,
            bench_iters=30,
            bench_warmup=5,
        )
        # Single candidate: the optimized kernel itself.
        history = loop.run(iter([("optimized", spec["candidate"])]))

        baseline_entry = loop._bench(spec["reference"])
        cand_entry = loop._bench(spec["candidate"])
        speedup = baseline_entry / cand_entry if cand_entry > 0 else 1.0

        table.add_row(
            kname, "PASS",
            f"{baseline_entry:.3f}", f"{cand_entry:.3f}",
            f"{speedup:.2f}x",
        )

    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
