# slipstream Architecture

Component-by-component design. This doc assumes you've read
[PLAN.md](PLAN.md) — that's the *why*, this is the *how*.

slipstream is **not** a kernel-generation system. The contribution is the
decode-optimized system around the kernels: paged FP8 KV cache, continuous
batcher, autotune cache, and the integration that turns those into a
runnable decode loop.

---

## High-level flow per engine step

```
1. Scheduler.step()
       → composes a SchedulerStep (decode tokens + prefill chunks)

2. Engine packs the step into model-input tensors
       → packed query Q [total_tokens, n_q_heads, head_dim]
       → block tables and seq_lens per sequence

3. Llama-3 forward:
       for each of 32 layers:
           Q,K,V = fp8_gemm(input, W_qkv_packed)              # decode-skinny GEMM
           Q     = rope(Q, position_ids)
           K,V   = rope(K, position_ids)
           append K,V to PagedKVCache  (FP8 quantized on write)
           ctx   = paged_attention(Q, kv_cache, block_tables) # crown-jewel kernel
           out   = fp8_gemm(ctx, W_o)
           h     = rmsnorm(out + residual)
           ...MLP...
       logits = lm_head(h)

4. Sampler picks next_token per sequence (argmax / top-k / top-p)

5. Engine calls scheduler.commit_output(decision, next_token)
       → scheduler reaps finished sequences, frees their KV blocks
```

Each phase has a single, well-defined responsibility. Components communicate
through plain Python dataclasses (no shared mutable state, no globals),
which makes them individually unit-testable.

---

## Paged KV cache (`slipstream.kvcache`)

The cache stores a global pool of fixed-size blocks. Each sequence holds
an ordered list of physical block ids (its *block table*). Layout per block::

    K:         [num_blocks, block_size, n_kv_heads, head_dim]   fp16 or fp8_e4m3
    V:         [num_blocks, block_size, n_kv_heads, head_dim]   same
    K_scales:  [num_blocks, block_size, n_kv_heads]             fp16  (FP8 only)
    V_scales:  [num_blocks, block_size, n_kv_heads]             fp16  (FP8 only)

Block size is **16** by default to match vLLM (so we can diff KV state
block-by-block against vLLM during correctness work). Per-token-per-head
scaling preserves accuracy on outlier tokens at < 1% storage overhead.

The attention kernel reads K/V directly from this layout — there is no
"materialize the contiguous view" step. The reference implementation in
`slipstream.attention.reference` does materialize (slow, simple, correct);
the Triton kernel walks blocks via the block table.

Allocator is a free-list with LIFO reuse (most recently freed = hottest
in L2 cache).

---

## FP8 quantization (`slipstream.kvcache.fp8`)

E4M3 (4-bit exponent, 3-bit mantissa). Range ±448.

```
scale  = max(|x|) / FP8_E4M3_MAX     (per scale-group)
x_fp8  = saturate(x / scale).cast(e4m3)
x_recover = x_fp8.cast(fp32) * scale
```

Per-group is the only knob:

- **Per-token-per-head** for KV cache (group = the `head_dim` vector of one
  token-head pair). One fp16 scale per `(token, head)`.
- **Per-token** for GEMM activations (group = the K-dim row). One scale
  per query token.
- **Per-channel** for GEMM weights (group = the K-dim column). Computed
  once offline; cached on disk.

E4M3 over E5M2 for everything: more mantissa, sufficient range. The Triton
kernel does the same math inline (dequant in fp32 registers; no extra HBM
round-trip).

---

## Continuous-batching scheduler (`slipstream.scheduler`)

The scheduler is the throughput story. Per `step()`:

1. **Decode pass.** For each in-flight sequence that's past prefill, schedule
   one decode token. Latency-sensitive — these go first.
2. **Prefill pass.** For each in-flight sequence still in prefill, schedule
   up to `chunk_size` prompt tokens. Capped per step to bound TTFT.
3. **Admission pass.** Drain the pending queue while KV capacity permits.

Admission gates on "enough free blocks for *one chunk*" rather than "enough
for the whole prompt" — the engine allocates blocks incrementally and the
scheduler naturally backpressures if the cache fills mid-prefill.

What's **not** implemented:

- **Preemption.** Production stacks (vLLM, SGLang) can swap KV blocks to
  host RAM to make room. We don't — pending requests wait. Documented
  failure mode rather than hidden complexity.
- **Priority queues.** First-in-first-out. Per-request priority is a
  one-line change when needed.

---

## Autotune cache (`slipstream.autotune`)

The cache is **the** scaling mechanism. There are no hand-tuned kernel
variants in slipstream. Each parameterized Triton template exposes a
`(BLOCK_M, BLOCK_N, num_warps, num_stages, ...)` config space; the
autotuner sweeps it across the decode-shape distribution and persists
winners.

Cache layout::

    slipstream/autotune/cache/gfx942.json     # MI300X
    slipstream/autotune/cache/gfx90a.json     # MI250X

Keys: `(kernel_id, shape_bucket)`. Shape bucketing rounds `(M, N, K, ctx,
batch)` to a discrete grid so trivially-different shapes share a tuned
config.

Discovery flow:

1. `slipstream-autotune --kernels all --shapes decode-suite` runs the
   sweep on the current GPU and writes winners to the cache.
2. At inference time, kernel entry points lookup their config from the
   cache. A miss logs a warning and falls back to a default config; a
   subsequent autotune fills the gap.

This is similar in spirit to TorchInductor's autotune cache, but specific
to slipstream's decode-shape distribution and keyed by GPU arch.

---

## Engine (`slipstream.engine`)

The engine ties the foundation pieces together. It owns:

- The `PagedKVCache`
- The `Scheduler`
- The Llama-3 forward callable
- A token sampler (argmax / top-k / top-p)

`Engine.step()` is the central loop:

```python
step = self.scheduler.step()
if step.is_empty():
    return 0
packed_q, block_tables, seq_lens = self._pack(step)
logits = self.model.forward(packed_q, self.cache, block_tables, seq_lens)
next_tokens = self.sampler.sample(logits)
self._commit(step, next_tokens)
```

The engine **does not** allocate per-step Python objects in the hot path
once running — `_pack` and `_commit` reuse persistent tensors. (Currently
trivial; this matters more once we measure.)

---

## Models (`slipstream.models.llama3`)

The Llama-3 forward path uses slipstream's primitives for every hot op:

- **Attention** → `slipstream.attention.paged_attention` (Phase 2 Triton)
- **GEMM** → `slipstream.gemm.fp8_gemm` (Phase 2 Triton)
- **RMSNorm** → `slipstream.kernels.rmsnorm` (Triton)
- **RoPE** → `slipstream.kernels.rope` (Triton)

Weights are loaded once from HuggingFace safetensors and optionally
FP8-quantized at load time (per-channel). Quantized weights are cached
to disk under `~/.cache/slipstream/weights/<model_id>/` so cold start
doesn't re-quantize.

---

## Benchmark harness (`slipstream.bench`)

Two layers:

- **Kernel benchmarks** (`bench/harness.py`) — compare one kernel function
  across eager / compile / slipstream. Used during kernel development.
- **Engine benchmarks** (`bench/runner.py`) — run a full decode workload
  through one or more baselines and compare. This is the published-numbers
  layer.

Baselines (`bench/baselines/`):

| Name | Backing stack | Hard dep? |
|---|---|---|
| `slipstream` | this project | always available |
| `vllm-rocm` | upstream vLLM (ROCm patches) | `pip install vllm` |
| `flash-attention-rocm` | CK `flash_attn_with_kvcache` | flash-attn ROCm wheel |
| `hipblaslt-fp16` | `torch @` on FP16 | `torch>=2.4` |
| `hipblaslt-fp8` | `torch._scaled_mm` | `torch>=2.4` |
| `torch-eager` | HF transformers eager forward | `pip install transformers` |
| `torch-compile` | `torch.compile(mode='max-autotune')` | `pip install transformers` |

All baselines implement `BaselineProtocol`. Missing-dep failures are caught
and reported as "skipped"; the run continues.

GPU-event timed, trimmed-mean over 30 runs after 10 warmup. Results dumped
as JSON for downstream plotting.

---

## Profiler (`slipstream.profiler`)

Wraps `torch.profiler` to capture per-kernel GPU timings while a model
runs a real forward pass. Two design choices:

- **Per-iteration normalization.** `key_averages()` returns cumulative
  time across the profiling window. We divide by `measured_iters` to get
  per-iteration; warmup iters are excluded (JIT, autotune, BLAS handle
  creation skew rankings).
- **Op-type aggregation** via `slipstream.profiler.classify`. Many GEMM
  variants (rocBLAS / hipBLASLt / CK tilings) all do the same thing —
  collapse them into op-type buckets so "optimize all matmul" is a single
  engineering project, not 12.

`AmdahlRanker` projects what 1.5× / 2× / 3× / 5× local speedups would
deliver end-to-end. This is how we choose what to optimize next.

---

## What's deliberately not implemented

| Not built | Why |
|---|---|
| Paged attention with non-contiguous query layout | Decode-only project; one Q token per sequence per step |
| KV cache compaction / preemption | Production stacks (vLLM, SGLang) do this; out of scope |
| Multi-GPU collectives | Single MI300X focus |
| Speculative decoding | Stretch goal; lands after §8 of [PLAN.md](PLAN.md) hits |
| LoRA / adapters | Stretch goal |
| HIP / CK custom kernels | Triton + autotune is the only kernel path |
| Hand-tuned per-shape variants | Anti-pattern; autotune does this |

---

## Performance methodology

See [BENCHMARKS.md](BENCHMARKS.md) for the protocol — GPU event timing,
trimmed mean, repeatability requirements, reproduction commands.
