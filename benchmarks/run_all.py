"""Reproduce all benchmark numbers in the README.

Run with:
    python benchmarks/run_all.py --iters 500 --output benchmarks/results/

Emits one JSON file per (kernel, dtype) tuple, plus a Markdown summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kvforge.bench.harness import BenchmarkHarness
from kvforge.bench.cli import bench_rmsnorm, bench_rope, bench_softmax
from kvforge.hardware import detect_gpu, has_triton


# Shape regimes per kernel: small / medium / large.
KERNEL_PLANS: dict[str, list[tuple]] = {
    "rmsnorm": [(4, 2048), (16, 4096), (32, 4096)],
    "rope": [(1, 4, 512, 64), (1, 8, 1024, 128), (2, 8, 2048, 128)],
    "softmax": [(4, 2048), (16, 4096), (32, 8192)],
}

BENCH_FNS = {"rmsnorm": bench_rmsnorm, "rope": bench_rope, "softmax": bench_softmax}


def _serialize_suite(suite) -> dict:
    """Convert a BenchmarkSuite into a JSON-friendly dict."""
    out = {"kernel": suite.kernel, "shape": list(suite.shape)}
    for label in ("eager", "compile", "kvforge"):
        r = getattr(suite, label, None)
        if r is None:
            out[label] = None
        else:
            out[label] = {
                "runtime_us": r.runtime_us,
                "dtype": str(r.dtype),
                "roofline": asdict(r.roofline) if r.roofline else None,
            }
    out["speedup_eager"] = suite.speedup("eager")
    out["speedup_compile"] = suite.speedup("compile")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"),
                        help="directory for JSON + Markdown output")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--dtypes", default="fp16",
                        help="comma-separated: fp16,bf16,fp32")
    parser.add_argument("--kernels", default="rmsnorm,rope,softmax")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    gpu = detect_gpu()
    print(f"Hardware: {gpu.name}  triton={has_triton()}")
    if not has_triton():
        print("WARNING: Triton unavailable; the 'kvforge' column will fall back to "
              "the eager reference. Run on a CUDA + Triton host for meaningful numbers.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    harness = BenchmarkHarness(gpu=gpu, warmup_iters=args.warmup, bench_iters=args.iters)

    all_results: list[dict] = []
    for dt_name in args.dtypes.split(","):
        dt_name = dt_name.strip()
        if dt_name not in dtype_map:
            print(f"unknown dtype {dt_name}; skipping")
            continue
        dtype = dtype_map[dt_name]
        if device.type == "cpu" and dtype != torch.float32:
            print(f"CPU fallback: dtype={dt_name} → fp32")
            dtype = torch.float32

        for kname in args.kernels.split(","):
            kname = kname.strip()
            if kname not in KERNEL_PLANS:
                continue
            for shape in KERNEL_PLANS[kname]:
                print(f"  [{dt_name}] {kname} shape={shape} ...", end=" ", flush=True)
                t0 = time.perf_counter()
                suite = BENCH_FNS[kname](harness, shape, dtype, device)
                dt = time.perf_counter() - t0
                rec = _serialize_suite(suite)
                rec["dtype_name"] = dt_name
                rec["bench_seconds"] = dt
                all_results.append(rec)
                print(f"done in {dt:.1f}s")

    # Write per-kernel JSON.
    out_json = args.output / "results.json"
    out_json.write_text(json.dumps({
        "gpu": {"name": gpu.name, "arch": gpu.arch,
                "peak_fp16_tflops": gpu.peak_fp16_tflops,
                "peak_bw_gb_s": gpu.peak_bw_gb_s},
        "config": {"iters": args.iters, "warmup": args.warmup},
        "results": all_results,
    }, indent=2))
    print(f"wrote {out_json}")

    # Write a Markdown summary.
    md_path = args.output / "summary.md"
    lines = [
        f"# Benchmark results — {gpu.name}",
        "",
        f"Iterations: {args.iters}, warmup: {args.warmup}, "
        f"trimmed mean (drop top/bottom 10%).",
        "",
        "| Kernel | Shape | Dtype | Eager (µs) | Compile (µs) | KVForge (µs) | "
        "vs Eager | vs Compile | % peak |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_results:
        eager = r["eager"]["runtime_us"] if r["eager"] else None
        comp = r["compile"]["runtime_us"] if r["compile"] else None
        kvf = r["kvforge"]["runtime_us"] if r["kvforge"] else None
        roof = r["kvforge"]["roofline"] if r["kvforge"] else None
        pct = roof["pct_of_peak"] * 100 if roof else None
        lines.append(
            f"| {r['kernel']} | {'x'.join(str(s) for s in r['shape'])} | {r['dtype_name']} | "
            f"{eager:.1f} | {comp:.1f if comp else 0} | {kvf:.1f} | "
            f"{r['speedup_eager']:.2f}× | "
            f"{(r['speedup_compile'] or 0):.2f}× | "
            f"{pct:.0f}%" if pct is not None else "-"
            + " |"
        )
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
