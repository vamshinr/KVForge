"""Persistent JSON cache for autotune results.

Cache layout on disk::

    slipstream/autotune/cache/
        gfx942.json           # MI300X
        gfx90a.json           # MI250X
        cpu.json              # for tests

Each file is::

    {
      "kernel_id": {
        "shape_bucket_string": {
          "config":          {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, ...},
          "measured_ms":     0.123,
          "triton_version":  "3.0.0",
          "recorded_at":     "2026-05-15T12:34:56Z",
          "src_hash":        "abc123..."     // optional, for invalidation
        },
        ...
      },
      ...
    }

The format is human-readable on purpose — easy to diff, copy across machines,
and audit when a kernel regresses.

Concurrency: read-mostly. The autotune CLI writes; at inference time we only
read. If two autotune runs race, the last writer wins; this is fine because
they should converge on the same winning config anyway.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Default cache directory — sibling to this file so an editable install
# carries the cache with it.
_DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


@dataclass(frozen=True)
class CacheKey:
    """Identifies one cache entry. Hashed by kernel + shape only — the arch
    selects the file, not the in-file key.
    """

    kernel_id: str          # e.g., "paged_attn_v1" or "fp8_gemm_decode"
    shape_bucket: str       # canonical string from shape_buckets.py

    def __str__(self) -> str:
        return f"{self.kernel_id}::{self.shape_bucket}"


@dataclass
class TunedConfig:
    """A single winning Triton config plus its provenance."""

    config: dict[str, Any]            # Triton kwargs (BLOCK_M etc.)
    measured_ms: float
    triton_version: str = ""
    recorded_at: str = ""
    src_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TunedConfig":
        return cls(
            config=data["config"],
            measured_ms=float(data["measured_ms"]),
            triton_version=data.get("triton_version", ""),
            recorded_at=data.get("recorded_at", ""),
            src_hash=data.get("src_hash", ""),
        )


class AutotuneCache:
    """Per-arch persistent cache.

    Parameters
    ----------
    arch:
        GPU arch tag — e.g. ``gfx942`` for MI300X. Determines the on-disk
        file. Use ``cpu`` for tests on CPU.
    cache_dir:
        Where to store the JSON files. Defaults to ``slipstream/autotune/cache``.
    """

    def __init__(self, arch: str, cache_dir: Path | str | None = None) -> None:
        self.arch = arch
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.cache_dir / f"{arch}.json"
        self._data: dict[str, dict[str, dict[str, Any]]] = self._load()

    # ---------- Public API ----------

    def lookup(self, key: CacheKey) -> TunedConfig | None:
        """Return the cached config for ``key``, or ``None`` if not present."""
        kdict = self._data.get(key.kernel_id, {})
        raw = kdict.get(key.shape_bucket)
        return TunedConfig.from_json(raw) if raw is not None else None

    def record(self, key: CacheKey, tuned: TunedConfig, persist: bool = True) -> None:
        """Insert or overwrite the cached config for ``key``."""
        kdict = self._data.setdefault(key.kernel_id, {})
        entry = tuned.to_json()
        # Stamp recorded_at if the caller didn't.
        if not entry.get("recorded_at"):
            entry["recorded_at"] = _dt.datetime.utcnow().isoformat() + "Z"
        kdict[key.shape_bucket] = entry
        if persist:
            self._save()

    def all_for_kernel(self, kernel_id: str) -> dict[str, TunedConfig]:
        kdict = self._data.get(kernel_id, {})
        return {sb: TunedConfig.from_json(v) for sb, v in kdict.items()}

    def kernels(self) -> list[str]:
        return list(self._data.keys())

    @property
    def path(self) -> Path:
        return self._path

    # ---------- Persistence ----------

    def _load(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # Don't crash on corrupted cache — start fresh, the next sweep fills it.
            return {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self._path)


@dataclass
class CacheStats:
    """Summary of cache coverage — used by the CLI's ``--report`` flag."""

    arch: str
    kernels: dict[str, int] = field(default_factory=dict)   # kernel_id -> entry count

    def total_entries(self) -> int:
        return sum(self.kernels.values())


def cache_stats(cache: AutotuneCache) -> CacheStats:
    stats = CacheStats(arch=cache.arch)
    for kernel_id in cache.kernels():
        stats.kernels[kernel_id] = len(cache.all_for_kernel(kernel_id))
    return stats
