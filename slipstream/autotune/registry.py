"""Registry of per-kernel autotuners.

Each Triton template registers a function::

    def tuner(cache: AutotuneCache, suite: str) -> None:
        # Sweep this kernel across `suite` and write winners into `cache`.

When a Triton template module is imported, it should append to
:data:`KERNEL_TUNERS`. Phase-2 kernels add themselves here when they land.

The registry is a plain dict — no fancy plugin machinery. Adding a tuner is::

    from slipstream.autotune.registry import KERNEL_TUNERS
    KERNEL_TUNERS["paged_attn_v1"] = paged_attn_tuner
"""

from __future__ import annotations

from collections.abc import Callable

from slipstream.autotune.cache import AutotuneCache


# kernel_id -> tuner_callable(cache, suite)
KERNEL_TUNERS: dict[str, Callable[[AutotuneCache, str], None]] = {}
