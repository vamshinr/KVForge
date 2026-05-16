"""``slipstream-bench`` — drive the cross-baseline benchmark suite.

The CLI selects baselines and a workload grid, runs everything, and writes
a JSON dump for downstream analysis. Use ``--out results.json`` to save
results; rerun with ``--plot results.json`` to produce the comparison
tables and roofline plots used in the README.

Examples
--------
Run the full production grid against all baselines::

    slipstream-bench --baselines all --workload production --out results.json

Just compare slipstream against vLLM-ROCm on the smoke grid::

    slipstream-bench --baselines slipstream vllm-rocm --workload smoke
"""

from __future__ import annotations

import argparse

from slipstream.bench.baselines import list_baselines
from slipstream.bench.runner import run_grid
from slipstream.bench.workloads import production_decode_grid, smoke_grid


_WORKLOAD_FACTORIES = {
    "production": production_decode_grid,
    "smoke": smoke_grid,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="slipstream-bench")
    p.add_argument("--baselines", nargs="+", default=["slipstream", "vllm-rocm"],
                   help=f"Subset of {list_baselines()}, or 'all'.")
    p.add_argument("--workload", default="production",
                   choices=list(_WORKLOAD_FACTORIES.keys()))
    p.add_argument("--model", default=None,
                   help="Override the workload's default model id.")
    p.add_argument("--out", default=None, help="JSON output path.")
    args = p.parse_args(argv)

    if args.baselines == ["all"]:
        baselines = list_baselines()
    else:
        baselines = args.baselines

    factory = _WORKLOAD_FACTORIES[args.workload]
    kwargs = {"model_id": args.model} if args.model else {}
    workloads = factory(**kwargs)

    run_grid(workloads, baselines, out_path=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
