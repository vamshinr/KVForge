"""Generic config-sweep driver: time each candidate config, keep the fastest.

Used by the CLI and by lazy on-demand tuning. The actual kernel launch is the
caller's job — they pass a ``run`` callable that takes a config dict and
returns measured milliseconds. This keeps the sweep logic kernel-agnostic.

Correctness: each candidate config is validated against the reference
implementation before timing. A config that fails correctness is rejected
(not timed). This is essential because Triton's autotune-internal correctness
check is lax; we use the same comparison the test suite does.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from slipstream.autotune.cache import AutotuneCache, CacheKey, TunedConfig


@dataclass
class SweepResult:
    """Outcome of a sweep over one (kernel, shape) point."""

    kernel_id: str
    shape_bucket: str
    winner: TunedConfig | None    # None = every candidate failed correctness
    n_candidates: int
    n_failed: int

    def succeeded(self) -> bool:
        return self.winner is not None


def sweep_configs(
    *,
    kernel_id: str,
    shape_bucket: str,
    candidates: Iterable[dict],
    run: Callable[[dict], float],
    verify: Callable[[dict], bool] | None = None,
    cache: AutotuneCache | None = None,
    triton_version: str = "",
    src_hash: str = "",
) -> SweepResult:
    """Sweep ``candidates``, return the fastest one that also passes verify.

    Parameters
    ----------
    kernel_id, shape_bucket:
        Cache key components.
    candidates:
        Iterable of config dicts. Each is passed to ``run`` and ``verify``.
    run:
        Callable that launches the kernel with a config and returns measured
        milliseconds (median over several iterations is recommended). Raises
        on failure — failures are caught and the config is skipped.
    verify:
        Optional callable that returns True if the config produces correct
        output (within tolerance). If omitted, no correctness gate is applied.
    cache:
        If provided, the winner is recorded.
    triton_version, src_hash:
        Provenance stamps recorded with the winner.
    """
    best_ms = math.inf
    best_cfg: dict | None = None
    n = 0
    failed = 0

    for cfg in candidates:
        n += 1
        try:
            if verify is not None and not verify(cfg):
                failed += 1
                continue
            ms = run(cfg)
        except Exception:
            # Out-of-resources, illegal config, etc. Triton's ROCm backend has
            # known issues with some (num_stages, num_warps) combos on gfx942.
            failed += 1
            continue
        if ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    winner = None
    if best_cfg is not None:
        winner = TunedConfig(
            config=best_cfg,
            measured_ms=best_ms,
            triton_version=triton_version,
            src_hash=src_hash,
        )
        if cache is not None:
            cache.record(CacheKey(kernel_id, shape_bucket), winner)

    return SweepResult(
        kernel_id=kernel_id,
        shape_bucket=shape_bucket,
        winner=winner,
        n_candidates=n,
        n_failed=failed,
    )
