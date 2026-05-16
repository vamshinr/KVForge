"""Cross-baseline benchmark runner.

Walks a workload grid and runs each (workload × baseline) combination. Each
result is recorded; baselines that fail to set up (missing dep, no GPU,
unsupported config) are logged but don't abort the run.

The runner is responsible for ordering — vLLM holds substantial HBM, so we
tear it down before launching the next baseline. Memory hygiene matters
because MI300X has 192GB but a Llama-3-8B + vLLM KV pool eats ~80GB on its
own.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from slipstream.bench.baselines import BaselineProtocol, make_baseline
from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult


def run_one(
    baseline: BaselineProtocol,
    inp: BenchmarkInput,
) -> DecodeResult | None:
    """Set up, run, and tear down a baseline. Returns ``None`` on failure."""
    try:
        baseline.setup(inp)
    except Exception as e:
        print(f"[{baseline.name}] setup failed: {e}")
        return None
    try:
        return baseline.run_workload(inp)
    except Exception as e:
        print(f"[{baseline.name}] run failed: {e}")
        traceback.print_exc()
        return None
    finally:
        try:
            baseline.teardown()
        except Exception as e:
            print(f"[{baseline.name}] teardown error (non-fatal): {e}")


def run_grid(
    workloads: list[BenchmarkInput],
    baseline_names: list[str],
    out_path: Path | str | None = None,
) -> list[DecodeResult]:
    """Run every (workload, baseline) pair. Returns the collected results.

    ``out_path`` (if given) gets a JSON dump of all results for downstream
    plotting / table generation.
    """
    results: list[DecodeResult] = []
    for inp in workloads:
        for name in baseline_names:
            print(f"== {name} @ batch={inp.batch_size} ctx={inp.prompt_len} "
                  f"fp8_kv={inp.fp8_kv} ==")
            t0 = time.perf_counter()
            try:
                baseline = make_baseline(name)
            except ImportError as e:
                print(f"  skipped ({e})")
                continue
            r = run_one(baseline, inp)
            if r is not None:
                results.append(r)
                print(f"  {r.tokens_per_sec:8.1f} tok/s   "
                      f"{r.ms_per_token_p50:7.3f} ms/tok p50   "
                      f"({time.perf_counter() - t0:.1f}s wall)")

    if out_path is not None:
        Path(out_path).write_text(json.dumps(
            [_serialize(r) for r in results], indent=2,
        ))
    return results


def _serialize(r: DecodeResult) -> dict:
    d = asdict(r)
    # BenchmarkInput is also a dataclass — already serialized via asdict.
    return d
