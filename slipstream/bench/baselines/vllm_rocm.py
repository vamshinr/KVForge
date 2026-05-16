"""vLLM-ROCm baseline adapter.

Wraps the upstream vLLM engine with the ROCm-side configuration that
matches our workload (no tensor parallel, no speculative, default scheduler).
We measure decode throughput by:

  1. Pre-feeding the same set of prompts to vLLM's ``LLM.generate`` once for
     warmup.
  2. Running the real measurement with ``LLM.generate`` and recording the
     per-step latency via vLLM's metrics output (or wall-clock fallback).

vLLM is a hard dependency only for this adapter — imported inside ``setup``.
Pin a known-good vLLM commit in the bench config (the API is still moving).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult


@dataclass
class VLLMRocmBaseline:
    """vLLM upstream w/ ROCm patches."""

    name: str = "vllm-rocm"
    # Pin to keep numbers comparable across runs; bump intentionally.
    pinned_vllm_commit: str = "v0.6.5"
    _llm: object | None = None

    def setup(self, inp: BenchmarkInput) -> None:
        try:
            from vllm import LLM, SamplingParams   # noqa: F401
        except ImportError as e:
            raise ImportError(
                "vllm is not installed. `pip install vllm` (ROCm build) to enable "
                "the vllm-rocm baseline."
            ) from e

        # vLLM's KV dtype option is "auto" | "fp8" | "fp8_e4m3" | "fp8_e5m2".
        kv_cache_dtype = "fp8" if inp.fp8_kv else "auto"
        self._llm = LLM(
            model=inp.model_id,
            dtype=inp.dtype,
            tensor_parallel_size=1,
            kv_cache_dtype=kv_cache_dtype,
            enforce_eager=False,
            disable_log_stats=True,
        )

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        from vllm import SamplingParams
        assert self._llm is not None, "call setup() first"

        prompts = _make_prompts(inp)
        sampling = SamplingParams(
            max_tokens=inp.decode_steps,
            ignore_eos=True,            # measure full decode_steps regardless of EOS
            temperature=0.0,
        )

        # Warmup.
        _ = self._llm.generate(prompts, sampling)

        # Measurement: full wall clock + (if available) vLLM's per-iter metrics.
        t0 = time.perf_counter()
        outputs = self._llm.generate(prompts, sampling)
        wall_s = time.perf_counter() - t0

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        # Wall-clock based metrics; per-step distribution requires vLLM's
        # built-in profiler hook (added in 0.6+). For now we report aggregate.
        ms_per_token_p50 = (wall_s * 1000) / max(total_tokens, 1)

        return DecodeResult(
            baseline=self.name,
            input=inp,
            ms_per_token_p50=ms_per_token_p50,
            ms_per_token_p99=ms_per_token_p50,   # placeholder until we wire in vLLM metrics
            tokens_per_sec=total_tokens / max(wall_s, 1e-6),
            ttft_ms_p50=0.0,                      # measured separately via streaming
            metadata={
                "vllm_commit_pin": self.pinned_vllm_commit,
                "kv_cache_dtype": "fp8" if inp.fp8_kv else "auto",
            },
        )

    def teardown(self) -> None:
        # vLLM holds a substantial KV pool; explicit drop to free HBM.
        self._llm = None
        import gc; gc.collect()


def _make_prompts(inp: BenchmarkInput) -> list[str]:
    """Synthetic prompts of approximately ``inp.prompt_len`` tokens.

    Real benchmarking uses the same tokenizer as the target model; this is
    a placeholder string-multiplier that gets within ~5% of the desired
    token length for English-ASCII inputs.
    """
    # "word " ≈ 1 token in typical LLama-3 tokenizers — close enough for sizing.
    base = "word " * inp.prompt_len
    return [base for _ in range(inp.batch_size)]
