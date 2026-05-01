"""CLI: `python -m kvforge.profile`."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from kvforge.hardware import detect_gpu
from kvforge.models.tinyllama import build_tinyllama, make_forward_fn
from kvforge.profiler.profile import ModelProfiler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kvforge-profile",
        description="Profile a model and rank its kernels by Amdahl impact.",
    )
    parser.add_argument("--model", default="tinyllama", choices=["tinyllama"],
                        help="model to profile (currently only built-in TinyLlama)")
    parser.add_argument("--context", type=int, default=512,
                        help="context length for prefill profiling")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--tiny", action="store_true",
                        help="use a tiny (vocab=1k, hidden=256) model for CPU smoke tests")
    parser.add_argument("--top", type=int, default=10,
                        help="number of top kernels to display")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="show raw per-kernel rows instead of aggregating by op type")
    args = parser.parse_args(argv)

    console = Console()
    gpu = detect_gpu()
    console.print(f"[bold cyan]Hardware:[/bold cyan] {gpu.name} "
                  f"({gpu.peak_fp16_tflops:.0f} TF FP16, {gpu.peak_bw_gb_s:.0f} GB/s)")

    model = build_tinyllama(tiny=args.tiny)
    forward_fn = make_forward_fn(
        batch_size=args.batch,
        seq_len=args.context,
        mode=args.mode,
    )

    profiler = ModelProfiler(warmup_iters=args.warmup, measured_iters=args.iters)
    console.print(f"[bold]Profiling[/bold] {args.model} "
                  f"(mode={args.mode}, ctx={args.context}, batch={args.batch})...")
    result = profiler.profile(model, forward_fn)

    table = Table(title=f"Top {args.top} kernels by Amdahl impact")
    table.add_column("#", justify="right")
    table.add_column("Kernel" if args.no_aggregate else "Op type")
    table.add_column("Time/iter (µs)", justify="right")
    table.add_column("Calls", justify="right")
    table.add_column("% of total", justify="right")
    table.add_column("Speedup @ 2x local", justify="right")
    table.add_column("Speedup @ 5x local", justify="right")

    rows = result.top_n(n=args.top, aggregated=not args.no_aggregate)
    for entry in rows:
        table.add_row(
            str(entry.rank),
            entry.op_type.value if not args.no_aggregate else entry.name[:50],
            f"{entry.total_us:.1f}",
            str(entry.call_count),
            f"{entry.fraction * 100:.1f}%",
            f"{entry.projections.get(2.0, 1.0):.2f}x",
            f"{entry.projections.get(5.0, 1.0):.2f}x",
        )

    console.print(table)
    console.print(f"[dim]Total GPU time per iter: {result.total_gpu_us:.1f} µs "
                  f"({result.measured_iters} iters averaged)[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
