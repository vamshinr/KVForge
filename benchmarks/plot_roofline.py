"""Plot a roofline diagram from benchmark results.

Run after `run_all.py`:
    python benchmarks/plot_roofline.py --input benchmarks/results/results.json

Generates `roofline.png` showing where each kernel sits on the roofline plot
relative to the GPU's compute and memory ceilings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib + numpy required. Install with: pip install matplotlib")
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("benchmarks/results/results.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/roofline.png"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    gpu_name = data["gpu"]["name"]
    peak_tflops = data["gpu"]["peak_fp16_tflops"]
    peak_bw = data["gpu"]["peak_bw_gb_s"]
    peak_gflops = peak_tflops * 1000.0
    ridge_point = peak_gflops / peak_bw

    fig, ax = plt.subplots(figsize=(9, 6))

    # Roofline ceilings: BW * AI for the slope, peak_gflops for the cap.
    ai_range = np.logspace(-1, 4, 200)
    bw_roof = peak_bw * ai_range
    ceiling = np.minimum(bw_roof, peak_gflops)
    ax.loglog(ai_range, ceiling, "k-", linewidth=2, label="Roofline ceiling")

    # Annotate ridge point.
    ax.axvline(ridge_point, color="gray", linestyle=":", alpha=0.5)
    ax.text(ridge_point, peak_gflops * 0.5,
            f"  ridge = {ridge_point:.1f} FLOP/B",
            verticalalignment="top", color="gray", fontsize=9)

    # Plot each kernel's measured point.
    colors = {"eager": "#888", "compile": "#3b82f6", "kvforge": "#22c55e"}
    markers = {"rmsnorm": "o", "rope": "s", "softmax": "^"}

    seen_labels: set[str] = set()
    for r in data["results"]:
        kernel = r["kernel"]
        marker = markers.get(kernel, "x")
        for label in ("eager", "compile", "kvforge"):
            entry = r.get(label)
            if not entry or not entry.get("roofline"):
                continue
            ai = entry["roofline"]["arithmetic_intensity"]
            gflops = entry["roofline"]["measured_gflops"]
            legend_label = f"{kernel} ({label})" if f"{kernel}-{label}" not in seen_labels else None
            seen_labels.add(f"{kernel}-{label}")
            ax.scatter(ai, gflops, c=colors[label], marker=marker, s=80,
                       edgecolors="black", linewidths=0.5, label=legend_label,
                       zorder=3)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"Roofline: {gpu_name}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
