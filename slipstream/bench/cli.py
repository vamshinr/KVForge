"""CLI: `python -m kvforge.bench`."""

from __future__ import annotations

import argparse
import sys

import torch
from rich.console import Console
from rich.table import Table

from kvforge.bench.harness import BenchmarkHarness
from kvforge.hardware import detect_gpu, has_triton
from kvforge.kernels.rmsnorm import (
    rmsnorm, rmsnorm_reference, rmsnorm_bytes, rmsnorm_flops,
)
from kvforge.kernels.rope import rope, rope_reference, rope_bytes, rope_flops
from kvforge.kernels.softmax import (
    softmax, softmax_reference, softmax_bytes, softmax_flops,
)


def _safe_compile(fn):
    """Try to compile `fn`; return None if compile fails (older torch / no GPU)."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.compile(fn, mode="max-autotune")
    except Exception:
        return None


def bench_rmsnorm(harness, shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    w = torch.randn(shape[-1], dtype=dtype, device=device)
    eager_fn = lambda: rmsnorm_reference(x, w)
    kvf_fn = lambda: rmsnorm(x, w)
    compiled = _safe_compile(rmsnorm_reference)
    compile_fn = (lambda: compiled(x, w)) if compiled is not None else None
    return harness.benchmark_three_way(
        "rmsnorm", shape, dtype, eager_fn, kvf_fn, compile_fn,
        flops=rmsnorm_flops(shape), bytes_moved=rmsnorm_bytes(shape, dtype),
    )


def bench_rope(harness, shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    s, d = shape[2], shape[3]
    angles = torch.arange(s, device=device).float()[:, None] * torch.arange(0, d, 2, device=device).float()
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)
    eager_fn = lambda: rope_reference(x, cos, sin)
    kvf_fn = lambda: rope(x, cos, sin)
    compiled = _safe_compile(rope_reference)
    compile_fn = (lambda: compiled(x, cos, sin)) if compiled is not None else None
    return harness.benchmark_three_way(
        "rope", shape, dtype, eager_fn, kvf_fn, compile_fn,
        flops=rope_flops(shape), bytes_moved=rope_bytes(shape, dtype),
    )


def bench_softmax(harness, shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    eager_fn = lambda: softmax_reference(x, dim=-1)
    kvf_fn = lambda: softmax(x, dim=-1)
    compiled = _safe_compile(softmax_reference)
    compile_fn = (lambda: compiled(x)) if compiled is not None else None
    return harness.benchmark_three_way(
        "softmax", shape, dtype, eager_fn, kvf_fn, compile_fn,
        flops=softmax_flops(shape), bytes_moved=softmax_bytes(shape, dtype),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvforge-bench")
    parser.add_argument("--kernels", default="rmsnorm,rope,softmax")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args(argv)

    console = Console()
    gpu = detect_gpu()
    console.print(f"[bold cyan]Hardware:[/bold cyan] {gpu.name}")
    console.print(f"[bold cyan]Triton available:[/bold cyan] {has_triton()}")
    console.print(f"[bold cyan]Peak FP16:[/bold cyan] {gpu.peak_fp16_tflops:.0f} TFLOPS  "
                  f"[bold cyan]Peak BW:[/bold cyan] {gpu.peak_bw_gb_s:.0f} GB/s")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype != torch.float32:
        console.print("[yellow]CPU fallback: forcing dtype=float32[/yellow]")
        dtype = torch.float32

    harness = BenchmarkHarness(gpu=gpu, warmup_iters=args.warmup, bench_iters=args.iters)

    table = Table(title=f"Benchmark results ({dtype})")
    table.add_column("Kernel")
    table.add_column("Shape")
    table.add_column("Eager (µs)", justify="right")
    table.add_column("Compile (µs)", justify="right")
    table.add_column("KVForge (µs)", justify="right")
    table.add_column("vs Eager", justify="right")
    table.add_column("vs Compile", justify="right")
    table.add_column("% peak", justify="right")

    plan = []
    for k in args.kernels.split(","):
        k = k.strip()
        if k == "rmsnorm":
            plan += [(bench_rmsnorm, (16, 4096)), (bench_rmsnorm, (32, 4096))]
        elif k == "rope":
            plan += [(bench_rope, (1, 8, 1024, 128)), (bench_rope, (2, 8, 2048, 128))]
        elif k == "softmax":
            plan += [(bench_softmax, (16, 4096)), (bench_softmax, (32, 8192))]

    for bench_fn, shape in plan:
        suite = bench_fn(harness, shape, dtype, device)

        def fmt(r):
            return f"{r.runtime_us:.1f}" if r else "-"

        speedup_eager = suite.speedup("eager")
        speedup_compile = suite.speedup("compile")
        pct_peak = suite.kvforge.roofline.pct_of_peak * 100 if suite.kvforge and suite.kvforge.roofline else None

        table.add_row(
            suite.kernel,
            "x".join(str(s) for s in suite.shape),
            fmt(suite.eager), fmt(suite.compile), fmt(suite.kvforge),
            f"{speedup_eager:.2f}x" if speedup_eager else "-",
            f"{speedup_compile:.2f}x" if speedup_compile else "-",
            f"{pct_peak:.0f}%" if pct_peak is not None else "-",
        )

    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
