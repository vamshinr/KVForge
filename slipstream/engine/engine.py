"""The slipstream decode engine.

Flow per ``step()``:

  1. Scheduler composes a batch (decode tokens + prefill chunks).
  2. Engine packs the batch into model-input tensors (one ``Q``, packed KVs).
  3. Model forward runs through all layers, producing logits.
  4. Token sampler picks the next token for each sequence.
  5. Engine appends the new K/V to the cache, then calls
     ``scheduler.commit_output`` for each decision.

The KV cache is sized at engine construction time using the available HBM
minus the model weights and activation budget.

Today this is the orchestration skeleton + a reference-implementation forward
that uses :func:`paged_attention_reference` and FP32 GEMM. The fast path
(Triton kernels) drops in via :mod:`slipstream.attention.triton_kernels` and
:mod:`slipstream.gemm.triton_kernels` once those land.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from slipstream.attention.reference import (
    paged_attention_reference,
    paged_attention_reference_prefill,
)
from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult
from slipstream.kvcache.paged import PagedKVCache
from slipstream.scheduler.continuous import Request, Scheduler


@dataclass
class EngineConfig:
    """Engine-wide knobs. Sensible defaults for an MI300X Llama-3-8B serving box."""

    # KV cache sizing.
    block_size: int = 16
    num_blocks: int = 4096           # 4096 blocks * 16 tokens = 65k tokens of cache
    kv_dtype_name: str = "fp16"      # "fp16" | "fp8" (E4M3)

    # Model sizing — Llama-3-8B defaults; the engine reads the real values
    # off the model config when one is loaded.
    n_layers: int = 32
    n_q_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden: int = 4096
    intermediate: int = 14336
    vocab_size: int = 128_256

    # Scheduler limits.
    max_seq_len: int = 8192
    max_batched_tokens: int = 4096
    max_concurrent_seqs: int = 256
    chunk_size: int = 512

    # Device.
    device: str = "cuda"


def _kv_dtype(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "fp8":  torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
    }[name]


class Engine:
    """Decode-optimized inference engine for MI300X.

    Parameters
    ----------
    config:
        Static config. The engine takes ownership of the KV cache and
        scheduler once constructed.
    model:
        Optional pre-loaded model (callable that takes packed query + cache
        + block tables and returns logits). When None, the engine runs in
        "reference-forward" mode where every layer goes through
        :func:`paged_attention_reference` and FP32 GEMM — correct but slow.

    Most users construct an engine via :meth:`from_model_id`.
    """

    def __init__(
        self,
        config: EngineConfig,
        model: object | None = None,
    ) -> None:
        self.config = config
        self.model = model

        self.cache = PagedKVCache(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            kv_dtype=_kv_dtype(config.kv_dtype_name),
            device=config.device,
        )
        self.scheduler = Scheduler(
            cache=self.cache,
            max_seq_len=config.max_seq_len,
            max_batched_tokens=config.max_batched_tokens,
            max_concurrent_seqs=config.max_concurrent_seqs,
            chunk_size=config.chunk_size,
        )

    # ---------- Public API ----------

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        *,
        dtype: str = "fp16",
        fp8_kv: bool = False,
        **overrides,
    ) -> "Engine":
        """Load a HF model id into an Engine. Implementation lands with the
        Llama-3 forward path; today this raises with the standard skip
        message used by the bench adapter.
        """
        raise NotImplementedError(
            "Engine.from_model_id will land with the Llama-3 forward "
            f"implementation (P3). Got model_id={model_id}, dtype={dtype}, "
            f"fp8_kv={fp8_kv}, overrides={overrides}"
        )

    def add_request(self, req: Request) -> None:
        self.scheduler.add(req)

    def step(self) -> int:
        """Execute one engine step. Returns the number of decisions processed.

        Returns 0 when the scheduler has no work — caller can break the loop.
        """
        step = self.scheduler.step()
        if step.is_empty():
            return 0
        # The model + token sampling happens here. In the reference mode the
        # engine still owns this loop, just with a slow forward. Production
        # Engine.run_step is overridden by the Llama-3 implementation (P3).
        self._reference_step(step)
        return len(step.decisions)

    def run_until_done(self, max_steps: int = 100_000) -> int:
        """Drain the request queue. Returns total steps executed."""
        for i in range(max_steps):
            if self.step() == 0:
                return i
        return max_steps

    def benchmark(self, inp: BenchmarkInput) -> DecodeResult:
        """Run a synthetic benchmark workload — exposes the engine to the
        bench runner. Reference mode is too slow for meaningful numbers;
        this raises until the fast forward lands.
        """
        raise NotImplementedError(
            "Engine.benchmark requires the fast (Triton) forward path. "
            f"Workload: {inp}"
        )

    # ---------- Internals ----------

    def _reference_step(self, step) -> None:
        """Reference-mode step. Used for correctness testing of the
        scheduler-engine integration, not perf.

        Every decision gets a no-op token; the scheduler is exercised but
        the math is a placeholder.
        """
        for d in step.decisions:
            # Allocate placeholder K/V (zeros) so the cache state stays
            # consistent with the scheduler's view.
            n_tokens = d.num_tokens
            shape = (n_tokens, self.config.n_kv_heads, self.config.head_dim)
            k = torch.zeros(shape, dtype=torch.float16, device=self.cache.device)
            v = torch.zeros(shape, dtype=torch.float16, device=self.cache.device)
            self.cache.append(d.request.seq_state, k, v)
            # Emit a placeholder token (the EOS token-id 0 by convention here).
            next_tok = 0 if not d.is_prefill or d.request.prompt_remaining == 0 else None
            self.scheduler.commit_output(d, next_tok)
