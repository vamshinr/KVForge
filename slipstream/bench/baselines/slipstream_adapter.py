"""Slipstream-as-baseline adapter.

This wires the slipstream engine (Phase-3 work) into the same baseline
protocol used by the other adapters, so the runner can compare side-by-side.
Until the engine lands, this stub raises with a clear message — the runner
treats that as "skipped, not failed."
"""

from __future__ import annotations

from dataclasses import dataclass

from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult


@dataclass
class SlipstreamBaseline:
    name: str = "slipstream"

    def setup(self, inp: BenchmarkInput) -> None:
        try:
            from slipstream.engine import Engine   # noqa: F401
        except ImportError as e:
            raise ImportError(
                "slipstream.engine is not yet implemented. The Phase-3 engine "
                "wires the parameterized Triton kernels into a runnable decode "
                "loop."
            ) from e

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        from slipstream.engine import Engine     # type: ignore
        engine = Engine.from_model_id(inp.model_id, dtype=inp.dtype, fp8_kv=inp.fp8_kv)
        return engine.benchmark(inp)

    def teardown(self) -> None:
        pass
