# slipstream — Project Plan

> A decode-optimized inference path for AMD Instinct MI300X.
> Fused paged attention. FP8 KV cache. MFMA-tuned. Continuous batching.
> Drop-in faster decode than vLLM-ROCm on Llama-3-class models.

---

## 1. Why this exists

LLM inference serving on MI300X is leaving 30–60% of peak throughput on the
floor today. The ROCm software stack — vLLM-ROCm, Flash-Attention-ROCm,
hipBLASLt — is **functional but substantially less tuned** than its CUDA
counterpart. Specifically:

- **Decode-step attention** on ROCm runs at 30–50% of MI300X's peak HBM
  bandwidth for production batch sizes (1–128). NVIDIA Flash-Attention-3
  routinely hits 70–80% on H100.
- **FP8 KV cache** is unsupported in ROCm's mainline serving path. MI300X has
  native FP8 (E4M3/E5M2) MFMA instructions sitting idle for KV storage.
- **Decode-shape skinny GEMM** (M ∈ {1..128}, N = K = hidden) is one of
  hipBLASLt's known weak spots — published numbers show 40–60% of peak on
  shapes that hit 80%+ on NVIDIA cuBLAS.

The opportunity isn't to invent a new algorithm or hand-write faster kernels.
**Hand-tuning doesn't scale**: one variant per shape per dtype per GPU gen,
and it's exactly what kernel-generation systems (Gimlet's kforge, Mirage,
KernelBench-style search) are designed to replace. We don't compete there.

Instead, we combine **five known-good ideas** into a ROCm-native, MI300X-tuned
decode path that no public project ships today, with the **system** as the
contribution:

1. vLLM-style **paged KV cache** with block tables
2. **Grouped-query attention** (Llama-3 / Mistral / Mixtral)
3. **FP8 KV cache** with per-block scales
4. **Continuous batching** of mixed prefill/decode
5. **Parameterized Triton templates + offline autotune cache** for the few
   kernels we own — never hand-tuned per shape

Where upstream is already strong (prefill GEMM via hipBLASLt, prefill
attention via FA-ROCm), we call it. Where we own a kernel (decode paged
attention with FP8 KV, FP8 decode-skinny GEMM), we write **one parameterized
template** and let the autotuner discover the right config per shape.

The headline claim, on landing, should be: *"On MI300X, Llama-3-8B decode is
N× faster end-to-end than vLLM-ROCm at batch 32, ctx 2048."* We will not
publish this number until it is real, reproducible, and honest.

---

## 2. What we are not building

To keep scope ruthless:

- **Not a serving stack.** No HTTP API, no tokenizer service, no metrics
  exporter, no multi-tenant. The deliverable is a library + a benchmark
  driver that loads HF weights and runs decode.
- **Not training.** Forward-only. No grads, no fused backward.
- **Not multi-GPU.** Single MI300X. Tensor/pipeline parallel is a separate
  discipline.
- **Not a kernel-generation agent.** Gimlet's kforge already does that.
  Our kernel work is parameterized templates + autotune, not codegen.
- **Not hand-tuned kernels.** No "rmsnorm_for_llama3_8b_bs32.py" variants.
  One template per op, autotuned over a config space, cached per
  `(shape_bucket, dtype, gfx_arch)` key.
- **Not prefill-optimized.** Prefill matters but is GEMM-dominated and
  already well-served by hipBLASLt + Flash-Attention-ROCm. Our wins come
  from decode.

---

## 3. Target hardware

**AMD Instinct MI300X (CDNA3, gfx942)**

| Spec | Value |
|---|---|
| Compute units | 304 |
| Peak FP16 / BF16 MFMA | 1307 TFLOPS |
| Peak FP8 MFMA | 2614 TFLOPS |
| HBM3 capacity | 192 GB |
| HBM3 bandwidth | 5325 GB/s |
| L2 cache | 16 MB |
| LDS per CU | 64 KB |
| Architecture | gfx942 |

Ridge point (FP16): 1307 / 5325 ≈ **245 FLOP/byte**.
Ridge point (FP8):  2614 / 5325 ≈ **491 FLOP/byte**.

Decode attention arithmetic intensity is ~2 FLOP/byte (dominated by KV cache
reads) → **deeply memory-bound**. FP8 KV halves the bytes → roughly doubles
the memory-bound ceiling. This is the central performance lever.

---

## 4. Baselines (the bar we must clear)

Every measurement reports our number side-by-side with these:

| Baseline | Stack | What it represents |
|---|---|---|
| **vLLM-ROCm** | vLLM upstream w/ ROCm patches | Current production SoTA on MI300X |
| **Flash-Attention-ROCm** | composable_kernel `mha_decode` | Best public attention kernel |
| **hipBLASLt FP16** | rocBLAS / hipBLASLt | Best public GEMM |
| **hipBLASLt FP8** | hipBLASLt + AMD scaling | FP8 GEMM SoTA on AMD |
| **torch.compile max-autotune** | TorchInductor → Triton | Generic compiler baseline |

We measure throughput (tok/s), per-token latency (ms/tok at p50/p99),
HBM bandwidth utilization (%), and MFMA utilization (%). All numbers are
trimmed-mean over 30 runs after 10 warmup iterations, GPU-event timed.

---

## 5. Phases

### Phase 0 — Foundation (this session, CPU-runnable)

- Rename `kvforge` → `slipstream`. Salvage profiler/Amdahl/roofline/harness.
- Project structure under `slipstream/`:
  - `attention/` — reference + Triton kernels
  - `gemm/` — reference + Triton kernels
  - `kvcache/` — paged allocator, block tables, FP8 quant
  - `scheduler/` — continuous batching
  - `engine/` — end-to-end decode driver
  - `models/` — Llama-3 in our style (GQA, RoPE, RMSNorm using our kernels)
  - `bench/` — baselines + runner
  - `profiler/` — kept (ROCm `torch.profiler` wrapper + Amdahl)
  - `kernels/` — utility ops (rmsnorm, rope, softmax) — demoted
  - `hardware.py` — MI300X spec
- Eager reference impls (CPU-runnable, ground truth):
  - Paged attention with GQA and FP8 KV
  - FP8 GEMM with per-tensor A / per-channel B scales
  - Continuous batching scheduler
- Benchmark harness scaffolding:
  - Baseline registry: vLLM-ROCm adapter, FA-ROCm adapter, hipBLASLt adapter,
    torch.compile adapter
  - GPU-event timer (existing) + workload definitions
- Tests for all reference impls (CPU, fast).

### Phase 1 — Autotune infrastructure + FP16 paged-attn template

- **Autotune cache** (`slipstream/autotune/`): persistent JSON keyed by
  `(kernel_id, shape_bucket, dtype, gfx_arch, triton_version)`. Records
  the winning config (BLOCK_*, num_warps, num_stages, etc.) + measured ms.
  Replayed at runtime; refreshed by `slipstream-autotune` CLI.
- **Shape bucketing**: round (M, N, K, ctx) to a discrete grid so we
  don't autotune for every micro-shape. Decode shapes cluster naturally —
  batch ∈ {1,2,4,8,16,32,64,128}, ctx ∈ powers of 2.
- **One** parameterized Triton template for paged-attn (FP16 K/V, GQA).
  All variation lives in `triton.autotune` configs, not in code forks.
- Correctness vs. reference within FP16 tolerance.
- Benchmark vs. FA-ROCm `mha_decode` and vLLM's `paged_attention_v2`.

**Gate to Phase 2:** within 15% of FA-ROCm on decode shapes after autotune.

### Phase 2 — FP8 KV cache (the crown jewel)

- Same single template, extended with an `FP8_KV: tl.constexpr` switch.
  No fork, no duplicate kernel — one source of truth.
- E4M3 KV storage with per-token-per-head scales.
- Quantize-on-write during prefill and decode-step KV append.
- Dequantize-on-read inside the attention kernel (no extra HBM round-trip).
- Numerics target: < 0.5% perplexity delta vs FP16 KV on WikiText-2.
- Benchmark: must beat FP16 vLLM by ≥ 1.5× at ctx ≥ 4096.

### Phase 3 — FP8 GEMM template for decode

- **One** parameterized Triton FP8 MFMA template covering `M ∈ {1..256}`,
  `N=K=hidden`. Autotuned over BLOCK_M, BLOCK_N, BLOCK_K, num_warps,
  num_stages, GROUP_M, SPLIT_K.
- Per-tensor A scale, per-channel B scale (hipBLASLt-compatible).
- Goal: beat hipBLASLt FP8 by ≥ 1.3× on skinny shapes (M ≤ 32);
  tie or call into hipBLASLt on square shapes.
- The autotuner picks SPLIT_K automatically — no manual switching.

### Phase 4 — End-to-end integration

- Llama-3-8B forward in `slipstream/models/llama3.py`. RoPE / RMSNorm /
  attention / GEMM all use our kernels.
- Continuous batcher feeds the model. Mixed prefill+decode batches.
- Load HF safetensors weights; quantize linear weights to FP8 ahead of time
  (offline, cached).

### Phase 5 — Win or document

- Full benchmark sweep, plotted curves.
- Where we win: document numbers honestly with reproduction commands.
- Where we lose: document the kernel, the shape, and the root cause.
  This is more credible than padding wins.

---

## 6. Critical design decisions

### KV cache layout

Each block holds `BLOCK_SIZE` tokens of K and V for a single sequence.
Block size is **16** to match vLLM (so we can cross-check correctness against
vLLM block-by-block). Layout per block:

```
K_block: [BLOCK_SIZE, n_kv_heads, head_dim]  fp8_e4m3
V_block: [BLOCK_SIZE, n_kv_heads, head_dim]  fp8_e4m3
K_scales: [BLOCK_SIZE, n_kv_heads]           fp16  (per-token, per-head)
V_scales: [BLOCK_SIZE, n_kv_heads]           fp16
```

Per-token scales (rather than per-block) cost a tiny bit more memory but
preserve accuracy on outlier tokens. We benchmark both.

### FP8 format choice

**E4M3 for KV.** Wider mantissa, sufficient range for post-softmax values
and K activations. E5M2 considered for V but loses too much precision on
the small-value tail.

**E4M3 for GEMM activations and weights.** Standard choice; matches
hipBLASLt's FP8 API.

### Continuous batching policy

- Decode tokens packed first (latency-sensitive).
- Prefill chunks filled into spare compute (throughput-friendly).
- Chunked prefill: max 512 tokens per chunk to keep TTFT bounded.
- Eviction: LRU on sequence slots, never mid-sequence.

### Why Triton over HIP/CK

- Triton + `@triton.autotune` lets the autotuner own the perf-tuning loop.
  Hand-HIP would force a per-shape rewrite cycle that doesn't scale.
- Triton's ROCm backend now lowers to MFMA cleanly (`tl.dot` → `v_mfma_*`).
- Where Triton leaves perf on the table, we **call the vendor library**
  (hipBLASLt, FA-ROCm) — we do not hand-write HIP. This keeps the project
  scalable as shapes/models/GPUs evolve.

### Autotune cache (the alternative to hand-tuning)

A single CLI: `slipstream-autotune --kernels all --shapes decode-suite`.
This sweeps each parameterized template across the shape distribution,
records the winning Triton config, and writes
`slipstream/autotune/cache/<gfx_arch>.json`. At inference time, kernels
look up their config from the cache. New shape → autotuner fills the gap
on first miss (with a warning). New GPU → re-run the autotune CLI. This
is the **only** "per shape tuning" knob in the project, and it's automated.

---

## 7. Salvaged from old slipstream

| Component | Why it stays | New location |
|---|---|---|
| `profiler/profile.py` | `torch.profiler` wrapper still useful | `slipstream/profiler/` |
| `profiler/amdahl.py` | Amdahl ranker is general | `slipstream/profiler/` |
| `profiler/classify.py` | Op-type classifier knows ROCm names | `slipstream/profiler/` |
| `optimizer/roofline.py` | Generic roofline math | `slipstream/roofline.py` |
| `optimizer/harness.py` | Correctness harness, generic | `slipstream/testing/harness.py` |
| `kernels/rmsnorm.py` | Used inside our Llama-3 forward | `slipstream/kernels/` |
| `kernels/rope.py` | Used inside attention pre-step | `slipstream/kernels/` |
| `kernels/softmax.py` | Used in unfused attention reference | `slipstream/kernels/` |
| `hardware.py` | MI300X spec | `slipstream/hardware.py` |

Removed: the agent-loop optimizer (`optimizer/search.py`). The new project is
hand-tuning, not search.

---

## 8. Success criteria

Concrete targets the project must hit for the README claims to be credible:

- [ ] Llama-3-8B decode, batch 32, ctx 2048: ≥ 1.7× tok/s vs vLLM-ROCm
- [ ] Llama-3-8B decode, batch 1, ctx 8192: ≥ 1.5× tok/s vs vLLM-ROCm
- [ ] Paged-attn kernel: ≥ 75% HBM bandwidth utilization on MI300X
- [ ] FP8 GEMM (M=32, N=K=4096): ≥ 1.3× hipBLASLt FP8
- [ ] FP8 KV cache: ≤ 0.5% perplexity delta on WikiText-2 vs FP16 KV
- [ ] All reference impls pass on CPU
- [ ] Reproducible: `slipstream-bench --suite full --out results.json` runs
  the whole sweep on a fresh checkout

If we miss any of these, the README says so explicitly with the actual
number we got.

---

## 9. Stretch goals (don't start until §8 hits)

- Speculative decoding (draft + verify in one engine call)
- Multi-LoRA decode (multiple adapters on one base model)
- Quantized linear weights (INT4 + FP8 activations, "mxfp4")
- Persistent kernel for decode (one launch per batch, not one per layer)
- Custom HIP kernel for the hottest decode-attn shape

---

## 10. Things that will go wrong (planning for it)

- **Triton-ROCm bugs.** The backend has known issues with some autotune
  configs (`num_stages > 2` is sometimes broken on gfx942). Workaround:
  pin known-good configs, file upstream issues.
- **vLLM-ROCm baseline moves.** Pin a vLLM commit hash in the bench config.
- **MFMA layout mismatches.** `tl.dot` returns row-major; some kernels
  expect column-major. Document the layout invariant per kernel.
- **FP8 underflow on near-zero KV values.** E4M3 min ~2e-3. Per-block scaling
  must clamp the dynamic range. Test with adversarial inputs.
