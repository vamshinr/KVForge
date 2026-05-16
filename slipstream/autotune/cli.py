"""``slipstream-autotune`` — refresh the autotune cache.

The CLI is a thin orchestrator: it picks the GPU arch, picks the shape suite
per kernel, dispatches the sweep, and prints a coverage report. The actual
sweep logic lives in :mod:`slipstream.autotune.sweep`; the per-kernel hooks
that know which Triton config space to enumerate live alongside each kernel
template (see ``slipstream.attention.triton_kernels`` and
``slipstream.gemm.triton_kernels`` in P2).

Usage::

    slipstream-autotune --kernels paged_attn fp8_gemm
    slipstream-autotune --kernels all --shapes decode-suite
    slipstream-autotune --report                  # show coverage only

This module currently includes only the orchestration scaffolding; the
kernel-specific tuners are wired up when the corresponding Triton templates
land in Phase 2.
"""

from __future__ import annotations

import argparse
import sys

from slipstream.autotune.cache import AutotuneCache, cache_stats
from slipstream.hardware import detect_gpu


def _gfx_arch_for_current_gpu() -> str:
    """Return the canonical arch tag used to name the cache file.

    Falls back to ``cpu`` when no GPU is present — useful for unit tests.
    """
    spec = detect_gpu()
    if spec.vendor == "amd":
        return {
            "cdna3": "gfx942",
            "cdna2": "gfx90a",
        }.get(spec.arch, spec.arch)
    if spec.vendor == "cpu":
        return "cpu"
    return spec.arch


def _cmd_report(cache: AutotuneCache) -> int:
    stats = cache_stats(cache)
    print(f"Autotune cache: {cache.path}")
    print(f"Arch: {cache.arch}")
    if not stats.kernels:
        print("  (empty)")
        return 0
    for kernel_id, n in stats.kernels.items():
        print(f"  {kernel_id:30s}  {n} entries")
    print(f"Total: {stats.total_entries()} entries")
    return 0


def _cmd_tune(cache: AutotuneCache, kernels: list[str], suite: str) -> int:
    """Dispatch per-kernel tuners.

    Each kernel template registers a tuner function (kernel_id -> callable);
    we look up and invoke each requested one. Until Phase-2 Triton templates
    land, this prints an informative stub.
    """
    from slipstream.autotune.registry import KERNEL_TUNERS

    if "all" in kernels:
        kernels = list(KERNEL_TUNERS.keys())
    if not kernels:
        print(
            "No kernel tuners registered yet. Phase-2 Triton templates will "
            "register themselves on import.",
            file=sys.stderr,
        )
        return 1

    for kernel_id in kernels:
        tuner = KERNEL_TUNERS.get(kernel_id)
        if tuner is None:
            print(f"unknown kernel: {kernel_id}", file=sys.stderr)
            continue
        print(f"-- Tuning {kernel_id} ({suite}) --")
        tuner(cache=cache, suite=suite)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="slipstream-autotune")
    p.add_argument("--kernels", nargs="*", default=["all"],
                   help="Kernel ids to tune (or 'all'). Use --report to skip tuning.")
    p.add_argument("--shapes", default="decode-suite",
                   help="Shape suite to sweep. Defaults to 'decode-suite'.")
    p.add_argument("--arch", default=None,
                   help="Override GPU arch tag. Useful for cross-machine cache work.")
    p.add_argument("--cache-dir", default=None,
                   help="Override the cache directory. Defaults to the package dir.")
    p.add_argument("--report", action="store_true",
                   help="Print cache coverage and exit (no tuning).")
    args = p.parse_args(argv)

    arch = args.arch or _gfx_arch_for_current_gpu()
    cache = AutotuneCache(arch=arch, cache_dir=args.cache_dir)

    if args.report:
        return _cmd_report(cache)
    return _cmd_tune(cache, args.kernels, args.shapes)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
