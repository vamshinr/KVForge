"""Flash-Attention-ROCm baseline — kernel-level only.

Exercises the Composable-Kernel ``mha_decode`` kernel directly, without a
serving stack around it. This is the right comparison point for the
*kernel* component of slipstream's attention story; end-to-end comparisons
go through the vLLM-ROCm adapter.

CK / flash-attn is imported lazily. As of FA-2.6 the ROCm path lives in
``flash_attn`` upstream with conditional builds; we pin to a tested release
via the env var ``SLIPSTREAM_FA_PIN``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult


@dataclass
class FlashAttentionRocmBaseline:
    name: str = "flash-attention-rocm"
    _ready: bool = False

    def setup(self, inp: BenchmarkInput) -> None:
        try:
            import flash_attn   # noqa: F401
        except ImportError as e:
            raise ImportError(
                "flash_attn (ROCm build) is not installed. See "
                "https://github.com/ROCmSoftwarePlatform/flash-attention "
                "for installation instructions."
            ) from e
        self._ready = True

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        """Decode-step microbenchmark.

        We *only* measure the attention kernel, not a full transformer pass.
        Inputs are synthetic; the comparison is to slipstream's attention
        kernel timed the same way.
        """
        import torch
        from flash_attn import flash_attn_with_kvcache    # type: ignore

        assert self._ready, "call setup() first"
        device = torch.device("cuda")
        # Llama-3-8B sizing fallback if the workload doesn't override.
        n_q_heads = 32
        n_kv_heads = 8
        head_dim = 128
        ctx = inp.prompt_len + inp.decode_steps

        torch.manual_seed(inp.seed)
        q = torch.randn(inp.batch_size, 1, n_q_heads, head_dim,
                        device=device, dtype=torch.float16)
        k_cache = torch.randn(inp.batch_size, ctx, n_kv_heads, head_dim,
                              device=device, dtype=torch.float16)
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.full(
            (inp.batch_size,), ctx - 1, device=device, dtype=torch.int32,
        )

        # Warmup.
        for _ in range(10):
            _ = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)
        torch.cuda.synchronize()

        # Per-step timing across `decode_steps` iterations.
        timings_ms: list[float] = []
        for _ in range(inp.decode_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens)
            end.record()
            end.synchronize()
            timings_ms.append(start.elapsed_time(end))

        timings_ms.sort()
        p50 = timings_ms[len(timings_ms) // 2]
        p99 = timings_ms[int(len(timings_ms) * 0.99)]
        # tokens/sec for the *whole batch* — one attention call decodes one
        # token per sequence simultaneously.
        toks_per_s = (inp.batch_size * 1000.0) / max(p50, 1e-6)

        return DecodeResult(
            baseline=self.name,
            input=inp,
            ms_per_token_p50=p50,
            ms_per_token_p99=p99,
            tokens_per_sec=toks_per_s,
            ttft_ms_p50=0.0,
            metadata={"kernel_only": True},
        )

    def teardown(self) -> None:
        self._ready = False
