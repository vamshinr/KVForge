"""Canonical workload grids for the benchmark suite.

The "production decode grid" is the matrix every baseline runs through. It
covers the production-relevant ranges of batch size and context length on
Llama-3-8B. The numbers we publish in the README are taken from this grid;
adding workloads here is the way to expand the comparison.
"""

from __future__ import annotations

from slipstream.bench.baselines.protocol import BenchmarkInput


LLAMA3_8B = "meta-llama/Meta-Llama-3-8B"


def production_decode_grid(
    model_id: str = LLAMA3_8B,
    dtype: str = "fp16",
) -> list[BenchmarkInput]:
    """The grid that drives the headline numbers.

    Each entry is one (batch, ctx) point at one fp8_kv setting. Latencies
    are reported per-token; throughput is reported per-second per-batch.
    """
    out: list[BenchmarkInput] = []
    for batch_size in [1, 8, 32, 128]:
        for prompt_len in [512, 2048, 8192]:
            for fp8_kv in [False, True]:
                out.append(BenchmarkInput(
                    model_id=model_id,
                    batch_size=batch_size,
                    prompt_len=prompt_len,
                    decode_steps=128,           # enough to amortize warmup
                    dtype=dtype,
                    fp8_kv=fp8_kv,
                ))
    return out


def smoke_grid(model_id: str = LLAMA3_8B) -> list[BenchmarkInput]:
    """A tiny grid for quick smoke tests during development."""
    return [
        BenchmarkInput(model_id=model_id, batch_size=1, prompt_len=128,
                       decode_steps=16, fp8_kv=False),
        BenchmarkInput(model_id=model_id, batch_size=8, prompt_len=512,
                       decode_steps=32, fp8_kv=True),
    ]
