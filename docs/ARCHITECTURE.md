# KVForge Architecture

This document explains the design of KVForge component-by-component. It assumes familiarity with LLM inference, GPU kernel programming, and basic compiler concepts.

## High-level data flow

KVForge runs in three phases. The user never moves between them manually — `kvforge-bench` invokes the whole pipeline; the CLIs are decomposed for debugging.

```
profile  →  rank by Amdahl  →  search loop  →  end-to-end verify
```

Each phase has a single, well-defined responsibility. The phases communicate through plain Python dataclasses (no shared mutable state, no globals), which makes the components individually unit-testable.

---

## Phase A: Profiling

**Module:** `kvforge.profiler`

The profiler wraps `torch.profiler` to capture per-CUDA-kernel timings while a model runs a real forward pass. Two design choices are worth flagging:

**Per-iteration normalization.** `torch.profiler.key_averages()` returns cumulative time across the whole profiling window. We divide by the configured `measured_iters` to get per-iteration time, which is what matters for Amdahl analysis. The `warmup_iters` are explicitly excluded — JIT compilation, autotuning, and one-time cuBLAS handle creation all happen during warmup and would skew rankings.

**Op-type aggregation.** Profilers expose dozens of cuBLAS GEMM variants (`ampere_sgemm_64x64_nn`, `cublasGemmEx`, etc.) that all do the same thing. The `AmdahlRanker` collapses these into op-type buckets via the classifier in `kvforge.profiler.classify`. This matters because optimizing "all matmul kernels" is a single engineering project, not 12 separate ones.

**Caveat:** The profiler is currently CUDA-only. On Apple Silicon or Intel GPU, it returns an empty kernel table (the CLI handles this gracefully). Adding MPS / SYCL profiling is straightforward but out of scope for v1.

---

## Amdahl ranking

**Module:** `kvforge.profiler.amdahl`

The Amdahl projection drives optimization priority. For a kernel that takes fraction `f` of total runtime, achieving local speedup `s` yields end-to-end speedup:

```
S_total = 1 / ((1 - f) + f / s)
```

Why this matters: a 5× speedup on a kernel that takes 5% of runtime yields only 1.04× end-to-end. The same effort spent on a 60% kernel yields 1.92×. The optimizer's job is to spend its budget where it compounds.

The ranker exposes `project()` for any speedup value, plus default projections at 1.5×, 2×, 3×, and 5×. These four anchors let the user eyeball the curve: if 2× and 5× both yield negligible end-to-end gain, the kernel isn't worth optimizing regardless of how successful the kernel-level work is.

---

## Phase B: The optimization loop

**Module:** `kvforge.optimizer`

The optimization loop is structurally identical to AutoKernel's: edit a single file, run the correctness harness, benchmark, keep or revert. KVForge's version is non-LLM (the candidate generator is a static iterator) which makes the loop deterministic and unit-testable.

### Correctness harness (`harness.py`)

Five stages, ordered cheapest first so failures abort early:

| Stage | What it catches | Typical runtime |
|---|---|---|
| Smoke | Compilation errors, gross numerical bugs | <1s |
| Shape sweep | Tile-remainder bugs, dtype-specific issues | 5–10s |
| Stability | Overflow, near-zero edge cases | 1–2s |
| Determinism | Race conditions in parallel reductions | <1s |
| Edge cases | Non-power-of-two boundary handling | 2–5s |

The shape sweep is the most important stage: it runs each candidate across 8+ shapes × 3 dtypes (24+ configurations). A kernel that works on `[16, 4096]` but breaks on `[7, 4097]` is broken — and that bug class is by far the most common failure mode in hand-written reduction kernels.

**Tolerance handling.** Different dtypes accumulate noise differently. The default tolerances (`fp32: 1e-4 atol`, `fp16: 1e-2 atol`, `bf16: 2e-2 atol`) match what cuBLAS/cuDNN typically achieve. They're conservative enough that real bugs always trip, loose enough that ordering-dependent reduction noise doesn't.

### Search loop (`search.py`)

The loop has three move-on criteria:

1. **Plateau:** 5 consecutive reverts → the candidate space is exhausted at this tier.
2. **Target:** `target_speedup` reached → spend the remaining budget elsewhere.
3. **Time budget:** wall-clock cap → graceful degradation.

These are borrowed verbatim from AutoKernel. The values (5 reverts, 2× target, 600s budget) are the defaults — production agent loops would tune them per kernel.

### Roofline analysis (`roofline.py`)

The roofline model classifies a kernel as compute-bound or memory-bound based on its arithmetic intensity (FLOPs per byte) compared to the GPU's compute-to-bandwidth ratio:

```
ridge_point = peak_FLOPS / peak_bandwidth
AI < ridge → memory-bound, optimize HBM traffic
AI > ridge → compute-bound, optimize FLOPS utilization
```

For an L4 GPU: `ridge ≈ 121 TF / 300 GB/s = 403 FLOP/byte`. RMSNorm has AI ≈ 1.5 FLOP/byte, so it's deep in memory-bound territory and 80% of peak bandwidth is the realistic ceiling. Matmul has AI > 1000 FLOP/byte — compute-bound, ceiling is peak FLOPS.

`recommend_tier()` translates roofline classification into a textual playbook entry. A real agent loop would feed these as prompts; in this static framework they appear in CLI output to help the user understand *why* a kernel is slow.

---

## Kernels

**Module:** `kvforge.kernels`

Each kernel is a self-contained module with three components:

1. **`<kernel>_reference()`** — eager PyTorch implementation. Always available. Used as the correctness oracle in tests and the harness.
2. **`<kernel>()`** — public API. Falls back to reference if Triton is unavailable or input is on CPU. This is what user code calls.
3. **`<kernel>_bytes()` / `<kernel>_flops()`** — roofline metadata. Used by the bench harness to compute percent-of-peak.

### Why Triton (and not CUDA C++)?

KVForge targets fast iteration over absolute peak performance:

- Triton compile time: 1–5 seconds. CUDA C++ via `load_inline`: 30+ seconds.
- The agent loop runs ~40 iterations/hour on Triton vs ~5/hour on CUDA C++.
- For memory-bound kernels (RMSNorm, RoPE, softmax), Triton routinely hits 80–95% of cuBLAS-equivalent throughput. The remaining gap isn't worth the iteration penalty.
- For compute-bound kernels (matmul), CUDA C++ has clear advantages — direct WMMA access, register-level control. KVForge's matmul story is "use cuBLAS" for v1; future work could add a CUDA C++ backend behind the same interface.

### Single-file invariant

In a real agent loop, each iteration touches exactly one kernel file. This is mechanical (the prompt enforces it), not architectural — but the project is structured to make it natural: each kernel lives in its own file, with no shared utility code.

---

## Phase C: End-to-end verification

The benchmark harness (`kvforge.bench.harness`) compares three baselines:

| Baseline | Implementation | Why it matters |
|---|---|---|
| Eager | PyTorch native (cuBLAS / ATen) | Lower bound — what users get out of the box |
| `torch.compile` | TorchInductor with `max-autotune` | Strong baseline — Inductor generates Triton kernels too |
| KVForge | Our hand-tuned Triton | The contribution |

Beating eager is easy. Beating `torch.compile` is the meaningful test — Inductor has a smart autotuner that often finds good Triton kernels automatically. Where KVForge wins, it's because:

1. Inductor's fusion is generic; ours is kernel-specific.
2. Inductor's autotuning has a wall-clock budget per kernel. Ours can spend hours on one kernel.
3. Some patterns (like the RoPE rotation) don't fuse cleanly under Inductor's pattern matcher.

The harness uses CUDA events (not `time.perf_counter`) for sub-microsecond accuracy and trimmed-mean aggregation (drop top/bottom 10%) to absorb scheduling noise.

---

## What's not implemented (deliberately)

These would all be reasonable extensions but are out of scope for v1:

- **Paged attention.** vLLM's PagedAttention is a layered abstraction over attention + KV cache; implementing it fully requires a serving stack. KVForge's attention kernel is the simpler "decode-step against contiguous KV cache" pattern.
- **Multi-GPU.** Single-device only. Distributed kernel work (NCCL collectives, pipeline parallel) is its own discipline.
- **Training kernels.** Inference and training have very different shape distributions (long-skinny matmuls, decode is memory-bound, etc.). The optimization playbook here is inference-specific.
- **A real LLM-driven agent.** The search loop accepts a static iterator of candidates. Wiring up an LLM that writes Triton code is a separate project.

These omissions are noted up-front in the README so reviewers don't have to discover them by reading the code.

---

## Performance methodology

See [BENCHMARKS.md](BENCHMARKS.md) for the full benchmark protocol, including:

- Hardware tested (L4, A10, RTX 3090)
- CUDA event timing methodology
- Trimmed-mean aggregation
- Correctness validation procedure (`pytest tests/ -v`)
- How to reproduce the headline numbers
