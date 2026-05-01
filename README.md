# KVForge

**Profile-guided kernel optimization for LLM inference.**

KVForge profiles a real LLM end-to-end, ranks inference-specific kernels by their contribution to total latency (Amdahl's law), and runs an iterative search loop that generates optimized Triton implementations gated by a five-stage correctness harness. Built to study where production LLM inference actually spends its time — and how to claw it back.

> **Status:** Research / portfolio project. Targets single-GPU inference of decoder-only transformers. Not a production serving system.

---

## Why this exists

Modern LLM inference is dominated by a small set of kernels: attention (paged or flash), RMSNorm, RoPE, fused softmax, and matmul. Vendor libraries cover matmul well, but the long tail of memory-bound kernels — and their interaction with KV cache layout — leaves significant performance on the table. Most existing kernel-search work (Korch, AutoKernel, KernelBench) treats kernels in isolation. KVForge instead:

1. **Starts from a real model** (TinyLlama-1.1B or Qwen2-0.5B) and profiles it with `torch.profiler` to get the actual kernel mix.
2. **Ranks kernels by Amdahl impact** so optimization effort goes where it compounds.
3. **Runs an iterative agent-style loop** (edit → correctness check → benchmark → keep/revert) over Triton candidates.
4. **Measures KV-cache-aware metrics** — prefill TTFT, decode tokens/sec, KV cache memory pressure under varying context lengths.

---

## Headline results

Measured on a single NVIDIA L4 (24GB), TinyLlama-1.1B, FP16, batch=1, context=2048.

| Kernel       | PyTorch eager | torch.compile | KVForge (Triton) | Speedup vs eager | % of peak BW |
|--------------|---------------|---------------|-------------------|------------------|--------------|
| RMSNorm      | 142 µs        | 99 µs         | **39 µs**         | **3.65×**        | 78%          |
| RoPE         | 211 µs        | 107 µs        | **84 µs**         | **2.51×**        | 71%          |
| Fused Softmax| 270 µs        | 330 µs        | **96 µs**         | **2.82×**        | 81%          |
| Decode attn  | 1.82 ms       | 1.41 ms       | **0.94 ms**       | **1.94×**        | 68%          |

**End-to-end:** 1.38× decode throughput improvement on TinyLlama-1.1B at context=2048.

> Numbers above are from the included benchmark suite. Re-run with `python -m kvforge.bench --model tinyllama --gpu auto`. See [BENCHMARKS.md](docs/BENCHMARKS.md) for full methodology, hardware variation, and correctness validation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase A: Profile                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Real LLM     │ -> │ torch.       │ -> │ Per-kernel   │       │
│  │ (TinyLlama)  │    │ profiler     │    │ time table   │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                          ┌───────▼───────┐       │
│                                          │ Amdahl ranker │       │
│                                          └───────┬───────┘       │
└──────────────────────────────────────────────────┼───────────────┘
                                                   │
┌──────────────────────────────────────────────────▼───────────────┐
│  Phase B: Search loop (per kernel)                               │
│                                                                  │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│   │ Candidate│->│  5-stage     │->│  Bench       │->│ keep /  │  │
│   │ generator│  │  correctness │  │  vs baseline │  │ revert  │  │
│   └────▲─────┘  └──────────────┘  └──────────────┘  └────┬────┘  │
│        │                                                 │       │
│        └─────────────────  history + roofline  ──────────┘       │
└──────────────────────────────────────────────────────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼───────────────┐
│  Phase C: End-to-end verification                                │
│  Replace nn.Module ops with optimized kernels, re-measure        │
│  TTFT, decode tokens/sec, output equivalence (cosine sim > 0.99) │
└──────────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component-by-component details.

---

## Quickstart

```bash
# Install
pip install -e ".[dev]"

# Profile a model and rank its kernels
python -m kvforge.profile --model tinyllama --context 2048

# Run the optimization loop on the top-3 kernels
python -m kvforge.optimize --kernels rmsnorm,rope,softmax --budget 50

# End-to-end benchmark vs baselines
python -m kvforge.bench --model tinyllama --compare eager,compile

# Run the test suite
pytest tests/ -v
```

---

## What's actually in this repo

| Path | What it does |
|---|---|
| `kvforge/profiler/` | `torch.profiler` wrapper, kernel classifier, Amdahl ranker |
| `kvforge/kernels/` | Triton implementations: rmsnorm, rope, softmax, paged attention |
| `kvforge/optimizer/` | Search loop, correctness harness, roofline calculator |
| `kvforge/bench/` | Benchmark harness with eager / `torch.compile` / KVForge comparison |
| `kvforge/models/` | Self-contained TinyLlama-style decoder for testing without HF deps |
| `tests/` | Numerical correctness, determinism, edge-case shape tests |
| `benchmarks/` | Reproducible scripts that emit the result tables in this README |
| `docs/` | Architecture writeup, benchmark methodology, design notes |

---

## Design choices worth flagging

- **Triton over CUDA C++.** Iteration speed matters more than absolute peak performance for this kind of search. Each candidate compiles in ~2 seconds; a CUDA C++ candidate takes 30+. The agent loop runs ~40 experiments/hour on Triton vs ~5/hour on CUDA C++.
- **Single-file kernel invariant.** Each candidate touches exactly one kernel file. Diffs stay small, reverts are clean (`git reset --hard`), and regressions are isolated.
- **Correctness gates *before* throughput.** A 5× speedup on a kernel that produces wrong outputs is worse than useless — it silently corrupts the model. Five stages: smoke test, shape sweep across 8 configs × 3 dtypes, numerical stability under adversarial inputs, determinism (3 runs bitwise identical), non-power-of-2 edges.
- **Roofline-guided tier selection.** The optimizer tags each kernel as compute-bound or memory-bound, then picks an optimization strategy from a tiered playbook (block sizes → memory access → compute → advanced). This is borrowed from AutoKernel; the novel piece here is applying it to inference-specific kernels with KV cache shape awareness.

---

## What this is not

- Not a vLLM or SGLang replacement. KVForge generates kernels; production serving needs continuous batching, paged attention scheduling, distributed routing, and a thousand other things.
- Not a multi-GPU framework. Single-device only.
- Not a training kernel optimizer. Inference shapes have very different optimization profiles than training (tall-skinny matmuls, decode is memory-bound, etc.).

---

## Influences and prior work

- **Korch** ([Hu et al., ASPLOS '24](https://arxiv.org/abs/2406.09465)) — operator fission and BLP-based kernel orchestration. KVForge borrows the idea of treating kernel selection as a search problem but applies it to inference-specific kernels rather than training graphs.
- **AutoKernel** ([Jaber & Jaber '26](https://arxiv.org/abs/2603.21331)) — iterative agent loop with correctness-gated benchmarking. KVForge's outer loop is structurally similar but extends the playbook with KV-cache-specific tiers.
- **FlashAttention** ([Dao et al., NeurIPS '22](https://arxiv.org/abs/2205.14135)) — tiled online softmax. KVForge's attention kernel uses this pattern.
- **vLLM PagedAttention** ([Kwon et al., SOSP '23](https://arxiv.org/abs/2309.06180)) — KV cache as pages. KVForge's KV layout is informed by but does not implement full paging.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

## Author

Vamshi Nagireddy — [LinkedIn](https://linkedin.com/in/vamshinr) · [GitHub](https://github.com/vamshinr) · [Blog](https://medium.com/@vamshire)
