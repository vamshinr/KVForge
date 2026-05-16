"""Common interface every baseline adapter implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkInput:
    """Canonical workload spec passed to every baseline.

    All baselines see the same shapes so comparisons are apples-to-apples.
    """

    model_id: str            # HF model id, e.g. "meta-llama/Meta-Llama-3-8B"
    batch_size: int          # number of concurrent sequences in decode
    prompt_len: int          # tokens of prefill per sequence (same for all in batch)
    decode_steps: int        # tokens to generate per sequence
    dtype: str = "fp16"      # model weight/activation dtype: "fp16" | "bf16"
    fp8_kv: bool = False     # whether to use FP8 KV cache (where supported)
    seed: int = 0            # deterministic prompts


@dataclass
class DecodeResult:
    """What the runner gets back from one baseline run."""

    baseline: str
    input: BenchmarkInput

    # Latency (per token, p50 / p99) in milliseconds.
    ms_per_token_p50: float
    ms_per_token_p99: float
    # Throughput aggregated across the batch.
    tokens_per_sec: float
    # Time-to-first-token (just prefill, p50) in milliseconds.
    ttft_ms_p50: float
    # Achieved HBM bandwidth and FLOPS (if measurable).
    hbm_bw_gb_s: float = 0.0
    measured_tflops: float = 0.0

    # Free-form metadata: baseline version, config, env.
    metadata: dict = field(default_factory=dict)


class BaselineProtocol(Protocol):
    """Interface every baseline implements.

    A baseline owns its full inference stack — KV cache, attention, GEMM,
    scheduler — and exposes a single entry point that produces decode-side
    metrics for a :class:`BenchmarkInput`.
    """

    name: str

    def setup(self, inp: BenchmarkInput) -> None:
        """Load model, allocate buffers, warm up. Called once per workload."""
        ...

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        """Execute the decode workload and return measured metrics."""
        ...

    def teardown(self) -> None:
        """Free model + buffers. Called once after all runs of a workload."""
        ...
