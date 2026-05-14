"""Single-GPU demo workload for KVForge.

Runs end-to-end in roughly 30-90 seconds on an AMD Instinct GPU
(MI300X / MI250X). Designed as the smallest convincing demonstration that the
profiler -> Amdahl ranker -> Triton kernel -> roofline pipeline works on your
hardware.

Phases:
  1. Detect hardware + report peak compute/bandwidth.
  2. Profile a small TinyLlama-shaped decoder in prefill mode with
     torch.profiler, then Amdahl-rank the kernels.
  3. Benchmark the optimized Triton RMSNorm against the eager reference at
     the actual hidden_size used by the model, and report roofline % of peak.

Run with:
    python scripts/demo.py                       # auto-detects GPU
    python scripts/demo.py --context 1024        # longer prefill
    python scripts/demo.py --hidden 4096         # benchmark at Llama-7B width
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow `python scripts/demo.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from kvforge.hardware import detect_gpu, has_triton, is_rocm
from kvforge.kernels.rmsnorm import rmsnorm, rmsnorm_bytes, rmsnorm_flops, rmsnorm_reference
from kvforge.models.tinyllama import build_tinyllama, make_forward_fn
from kvforge.optimizer.roofline import RooflineCalculator
from kvforge.profiler.profile import ModelProfiler


def _bench_ms(fn, warmup: int = 10, iters: int = 100) -> float:
    """Trimmed-mean wall-time in milliseconds for a no-arg callable."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    if torch.cuda.is_available():
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return sum(trimmed) / len(trimmed)


def phase_hardware(console: Console):
    gpu = detect_gpu()
    rocm = is_rocm()
    triton_ok = has_triton()

    table = Table(title="Hardware", show_header=False, box=None)
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Device",           gpu.name)
    table.add_row("Arch / vendor",    f"{gpu.arch} ({gpu.vendor})")
    table.add_row("Peak FP16",        f"{gpu.peak_fp16_tflops:.0f} TFLOPS")
    table.add_row("Peak FP32",        f"{gpu.peak_fp32_tflops:.0f} TFLOPS")
    table.add_row("Peak BW",          f"{gpu.peak_bw_gb_s:.0f} GB/s")
    table.add_row("CUs / SMs",        str(gpu.sm_count))
    table.add_row("PyTorch backend",  "ROCm" if rocm else ("GPU" if torch.cuda.is_available() else "CPU"))
    table.add_row("Triton available", "yes" if triton_ok else "no")
    console.print(table)
    console.print()
    return gpu, triton_ok


def phase_profile(console: Console, batch: int, context: int, n_layers: int) -> None:
    console.rule("[bold]Phase 2 — profile TinyLlama prefill")
    model = build_tinyllama(n_layers=n_layers)
    forward_fn = make_forward_fn(batch_size=batch, seq_len=context, mode="prefill")

    profiler = ModelProfiler(warmup_iters=3, measured_iters=8)
    console.print(f"Profiling batch={batch} context={context} layers={n_layers} ...")
    result = profiler.profile(model, forward_fn)
    console.print(f"Total GPU time per iter: [bold]{result.total_gpu_us / 1000:.2f} ms[/bold] "
                  f"({result.measured_iters} iters averaged)\n")

    table = Table(title="Top kernels by Amdahl impact (aggregated by op type)")
    table.add_column("#", justify="right")
    table.add_column("Op type")
    table.add_column("µs/iter", justify="right")
    table.add_column("% of total", justify="right")
    table.add_column("S_total @ 2× local", justify="right")
    table.add_column("S_total @ 5× local", justify="right")
    for entry in result.top_n(n=8, aggregated=True):
        table.add_row(
            str(entry.rank),
            entry.op_type.value,
            f"{entry.total_us:.1f}",
            f"{entry.fraction * 100:.1f}%",
            f"{entry.projections.get(2.0, 1.0):.2f}×",
            f"{entry.projections.get(5.0, 1.0):.2f}×",
        )
    console.print(table)
    console.print()


def phase_rmsnorm_bench(console: Console, gpu, n_rows: int, hidden: int, dtype: torch.dtype) -> None:
    console.rule("[bold]Phase 3 — RMSNorm: eager vs Triton + roofline")
    if not torch.cuda.is_available():
        console.print("[yellow]No GPU device available — skipping kernel benchmark.[/yellow]")
        return

    device = torch.device("cuda")
    x = torch.randn(n_rows, hidden, dtype=dtype, device=device)
    w = torch.randn(hidden, dtype=dtype, device=device)

    # Correctness sanity check before we trust the speedup number.
    out_ref = rmsnorm_reference(x, w)
    out_kvf = rmsnorm(x, w)
    max_diff = (out_ref.float() - out_kvf.float()).abs().max().item()
    tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    ok = max_diff < tol
    console.print(f"Correctness check: max |Δ| = {max_diff:.2e}  "
                  f"({'PASS' if ok else 'FAIL'}, tol={tol:.0e})")
    if not ok:
        console.print("[red]Numerical mismatch — refusing to report a speedup.[/red]")
        return

    eager_ms = _bench_ms(lambda: rmsnorm_reference(x, w))
    kvf_ms = _bench_ms(lambda: rmsnorm(x, w))

    shape = (n_rows, hidden)
    flops = rmsnorm_flops(shape)
    bytes_moved = rmsnorm_bytes(shape, dtype)
    roof = RooflineCalculator(gpu, dtype_is_fp16=(dtype in (torch.float16, torch.bfloat16)))
    r_eager = roof.analyze(flops, bytes_moved, eager_ms / 1000)
    r_kvf = roof.analyze(flops, bytes_moved, kvf_ms / 1000)

    table = Table(title=f"RMSNorm @ shape={shape}, dtype={str(dtype).split('.')[-1]}")
    table.add_column("Variant")
    table.add_column("Latency (µs)", justify="right")
    table.add_column("Achieved BW (GB/s)", justify="right")
    table.add_column("% of peak BW", justify="right")
    table.add_row("eager (PyTorch)",
                  f"{eager_ms * 1000:.1f}",
                  f"{r_eager.measured_bw_gb_s:.0f}",
                  f"{(r_eager.measured_bw_gb_s / gpu.peak_bw_gb_s) * 100:.1f}%")
    table.add_row("kvforge (Triton)",
                  f"{kvf_ms * 1000:.1f}",
                  f"{r_kvf.measured_bw_gb_s:.0f}",
                  f"{(r_kvf.measured_bw_gb_s / gpu.peak_bw_gb_s) * 100:.1f}%")
    console.print(table)
    console.print(f"Speedup vs eager: [bold green]{eager_ms / kvf_ms:.2f}×[/bold green]")
    console.print(f"Arithmetic intensity: {r_kvf.arithmetic_intensity:.2f} FLOP/B → "
                  f"{r_kvf.bound}-bound (ridge point {roof.ridge_point:.0f} FLOP/B)")


def main() -> int:
    parser = argparse.ArgumentParser(description="KVForge end-to-end demo.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=512,
                        help="prefill sequence length")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="TinyLlama decoder depth")
    parser.add_argument("--hidden", type=int, default=2048,
                        help="rmsnorm hidden dim for the kernel benchmark")
    parser.add_argument("--rows", type=int, default=4096,
                        help="rmsnorm batch*seq rows for the kernel benchmark")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    console = Console()
    console.rule("[bold]KVForge demo")
    gpu, _triton = phase_hardware(console)
    phase_profile(console, batch=args.batch, context=args.context, n_layers=args.n_layers)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    phase_rmsnorm_bench(console, gpu, n_rows=args.rows, hidden=args.hidden, dtype=dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
