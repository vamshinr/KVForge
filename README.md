# slipstream

**A decode-optimized LLM inference path for AMD Instinct MI300X.**

Paged FP8 KV cache. Continuous batching. Autotuned Triton kernels.
Drop-in faster decode than vLLM-ROCm on Llama-3-class models.

> **Status:** Foundation complete (paged KV cache, FP8 quantization, paged
> attention reference, continuous batching scheduler, FP8 GEMM reference,
> autotune cache, benchmark harness, baseline adapters). Triton kernel
> templates and the Llama-3 forward path are next.
> See [docs/PLAN.md](docs/PLAN.md) for the multi-phase plan.

---

## Why this exists

LLM inference serving on MI300X is leaving 30–60% of peak throughput on the
floor today. The ROCm stack — vLLM-ROCm, Flash-Attention-ROCm, hipBLASLt —
is **functional but substantially less tuned** than its CUDA counterpart.
Specifically:

- **Decode-step attention** on ROCm runs at 30–50% of MI300X's peak HBM
  bandwidth for production batch sizes (1–128). NVIDIA Flash-Attention-3
  routinely hits 70–80% on H100.
- **FP8 KV cache** is unsupported in ROCm's mainline serving path. MI300X
  has native FP8 MFMA instructions sitting idle for KV storage.
- **Decode-shape skinny GEMM** (M ∈ {1..128}, N = K = hidden) is a known
  hipBLASLt weak point — 40–60% of peak on shapes that hit 80%+ on cuBLAS.

slipstream is the **system** that combines five known-good ideas into a
ROCm-native decode path that no public project ships today:

1. vLLM-style **paged KV cache** with block tables
2. **Grouped-query attention** (Llama-3 / Mistral / Mixtral)
3. **FP8 KV cache** with per-token, per-head scales
4. **Continuous batching** of mixed prefill/decode
5. **Parameterized Triton templates** with offline autotune — *never* hand-
   tuned per shape

Where upstream is already strong (prefill GEMM via hipBLASLt, prefill
attention via FA-ROCm), we call it. Where we own a kernel (decode paged
attention with FP8 KV, FP8 decode-skinny GEMM), we write **one parameterized
template** and let the autotuner discover the right config per shape.

---

## Why not just hand-tune kernels?

Hand-tuning doesn't scale: one variant per shape per dtype per GPU generation.
It's also exactly what automated kernel-generation systems (Gimlet's kforge,
Mirage, KernelBench) exist to replace. slipstream is **structurally
complementary** to those systems — it's the layer that consumes optimized
kernels and produces faster end-to-end inference, not another kernel
optimizer.

If you have a kernel-generation system, slipstream is what runs the kernels.
If you don't, slipstream's autotuner sweeps Triton's config space and caches
the winner per `(shape_bucket, dtype, gfx_arch)` key.

---

## Headline results

*Coming after the Triton kernels and Llama-3 forward land. We will not
publish numbers we can't reproduce on demand.*

Success criteria the project must hit before publishing:

- Llama-3-8B decode, batch 32, ctx 2048: **≥ 1.7×** tok/s vs vLLM-ROCm
- Llama-3-8B decode, batch 1, ctx 8192: **≥ 1.5×** tok/s vs vLLM-ROCm
- Paged-attn kernel: **≥ 75% HBM bandwidth** utilization on MI300X
- FP8 GEMM (M=32, N=K=4096): **≥ 1.3×** hipBLASLt FP8
- FP8 KV cache: **≤ 0.5% perplexity delta** on WikiText-2 vs FP16 KV

If we miss any of these, the README says so explicitly with the actual
number we got.

---

## Architecture

```
                          ┌──────────────────────┐
                          │   slipstream-serve   │
                          │  (CLI / lib entry)   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │       Engine         │   ◀── slipstream/engine
                          │ orchestrates a step  │
                          └──────────┬───────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
        ┌───────▼──────┐     ┌───────▼───────┐    ┌───────▼─────┐
        │  Scheduler   │     │ Llama-3 model │    │  PagedKV    │
        │  continuous  │     │ (uses our     │    │  cache +    │
        │  batching    │     │  primitives)  │    │  FP8 quant  │
        └──────────────┘     └───────┬───────┘    └─────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
        ┌───────▼──────┐     ┌───────▼───────┐    ┌───────▼─────┐
        │  Paged-attn  │     │  FP8 GEMM     │    │ RMSNorm /   │
        │  Triton tpl  │     │  Triton tpl   │    │ RoPE Triton │
        │  (autotuned) │     │  (autotuned)  │    │ utility ops │
        └───────┬──────┘     └───────┬───────┘    └─────────────┘
                │                    │
                └──────┬─────────────┘
                       │
                ┌──────▼──────┐
                │  Autotune   │   ◀── slipstream/autotune
                │  cache JSON │       (replays winning configs)
                └─────────────┘
```

Component-level docs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Multi-phase plan: [docs/PLAN.md](docs/PLAN.md).

---

## Comparison points

`slipstream-bench` benchmarks against every relevant baseline on the same
workload grid:

| Baseline | What it represents |
|---|---|
| **vLLM-ROCm** | Current production SoTA on MI300X |
| **Flash-Attention-ROCm** | Best public attention kernel (CK `mha_decode`) |
| **hipBLASLt FP16 / FP8** | Best public GEMM on AMD |
| **torch.compile max-autotune** | Generic compiler baseline |

All numbers are GPU-event timed, trimmed-mean over 30 runs after 10 warmup
iterations.

---

## Install

```bash
pip install -e ".[triton,dev]"
```

Optional baseline dependencies (only needed for the comparisons you actually
run):

```bash
pip install vllm transformers accelerate           # vllm-rocm + torch baselines
pip install flash-attn --no-build-isolation         # flash-attention-rocm
```

---

## Use

Profile a model:

```bash
slipstream-profile --model meta-llama/Meta-Llama-3-8B \
    --mode decode --batch 8 --context 2048
```

Run autotune across the production shape grid:

```bash
slipstream-autotune --kernels all --shapes decode-suite
slipstream-autotune --report
```

Benchmark slipstream vs every baseline:

```bash
slipstream-bench --baselines all --workload production --out results.json
```

Run a tiny end-to-end with the engine (when ready):

```bash
slipstream-serve --model meta-llama/Meta-Llama-3-8B \
    --prompts-file prompts.txt --max-tokens 128 --fp8-kv
```

---

## What's deliberately not in scope

| Not built | Why |
|---|---|
| HTTP serving stack | Use vLLM or SGLang on top |
| Multi-GPU (TP/PP) | Single-MI300X focus; another discipline |
| Training kernels | Inference workload distribution is different |
| Kernel-generation agent | Gimlet's kforge and others already do this |
| Hand-tuned kernels per shape | Doesn't scale — autotune does it instead |
| Prefill optimization | Already well-served by hipBLASLt + FA-ROCm |

---

## Repository layout

```
slipstream/
├── attention/        Paged-attn reference + Triton template (Phase 2)
├── gemm/             FP8 GEMM reference + Triton template (Phase 2)
├── kvcache/          Paged cache, block table, FP8 quant/dequant
├── scheduler/        Continuous batching
├── engine/           End-to-end decode driver
├── models/           Llama-3 forward path (Phase 3)
├── autotune/         Persistent autotune cache + sweep driver
├── bench/            Cross-baseline benchmark harness
├── profiler/         torch.profiler wrapper + Amdahl ranker
├── kernels/          Utility ops (RMSNorm, RoPE, softmax) — Triton
├── testing/          Correctness harness
├── hardware.py       GPU specs (MI300X, MI250X, ...)
└── roofline.py       Roofline math
docs/
├── PLAN.md           The multi-phase project plan (read this first)
├── ARCHITECTURE.md   Component-by-component
├── DESIGN_DECISIONS.md
└── BENCHMARKS.md     Reproduction protocol
tests/                Unit tests — all CPU-runnable
```

---

## License

Apache-2.0.
