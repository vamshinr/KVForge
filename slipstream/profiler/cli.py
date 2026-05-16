"""``slipstream-profile`` — rank a model's kernels by Amdahl impact.

Loads a HuggingFace model and runs ``torch.profiler`` over a few iterations
of prefill or decode, then prints the kernels in descending order of total
GPU time. The Amdahl projection shows the end-to-end speedup we'd see if
each kernel got 2x or 5x faster — answers "where would optimization effort
actually pay off."

Use this to *audit* a model before committing to kernel work. If the top
kernels are all already optimized vendor code (hipBLASLt, FA-ROCm) and the
remaining 5% is what's slow, slipstream's contribution can't move the
needle on that model and we should pick a different target.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from slipstream.hardware import detect_gpu
from slipstream.profiler.profile import ModelProfiler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="slipstream-profile",
        description="Profile a HuggingFace model and rank its kernels by Amdahl impact.",
    )
    parser.add_argument("--model", required=True,
                        help="HF model id, e.g. meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--context", type=int, default=2048,
                        help="context length for the profile run")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--top", type=int, default=15,
                        help="number of top kernels to display")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="show raw per-kernel rows instead of aggregating by op type")
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print(f"transformers is required for --model: {e}", file=sys.stderr)
        return 2

    console = Console()
    gpu = detect_gpu()
    console.print(f"[bold cyan]Hardware:[/bold cyan] {gpu.name} "
                  f"({gpu.peak_fp16_tflops:.0f} TF FP16, {gpu.peak_bw_gb_s:.0f} GB/s)")

    console.print(f"[bold]Loading[/bold] {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to("cuda").eval()

    def forward_fn():
        if args.mode == "prefill":
            ids = torch.randint(0, 1000, (args.batch, args.context),
                                device="cuda", dtype=torch.long)
            return model(input_ids=ids).logits
        else:
            ids = torch.randint(0, 1000, (args.batch, 1), device="cuda", dtype=torch.long)
            return model(input_ids=ids).logits

    profiler = ModelProfiler(warmup_iters=args.warmup, measured_iters=args.iters)
    console.print(f"[bold]Profiling[/bold] (mode={args.mode}, ctx={args.context}, "
                  f"batch={args.batch})...")
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
