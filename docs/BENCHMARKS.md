# Benchmarks: Methodology and Results

This document describes how slipstream's benchmarks are run, what they measure, and how to reproduce them.

> **Status:** Methodology is locked in; numbers land after the Triton kernels
> and Llama-3 forward path finish (see [PLAN.md](PLAN.md), Phases 2–3).
> Until then, this doc is the protocol reference, not a results page.

---

## TL;DR — how to run

```bash
# Full cross-baseline production grid (Llama-3-8B decode, MI300X)
slipstream-bench --baselines all --workload production --out results.json

# Smoke grid (a few shapes) — for development
slipstream-bench --baselines slipstream vllm-rocm --workload smoke

# Just one baseline against another
slipstream-bench --baselines slipstream flash-attention-rocm --workload smoke
```

All baselines write `DecodeResult` rows to `results.json`. The plotting/
table-generation utilities are in `slipstream/bench/report.py` (lands with
the first published number).

---

## What we measure

| Metric | Definition |
|---|---|
| **tokens/sec** | Total tokens generated (across the batch) per wall-clock second of decode time |
| **ms/token p50** | Median per-token latency, measured at the engine level (one decode step from input to next-token id) |
| **ms/token p99** | 99th percentile — captures tail latency relevant for serving SLOs |
| **TTFT p50** | Median time-to-first-token (prefill latency) |
| **HBM BW %** | Measured DRAM bytes moved / peak HBM bandwidth (5325 GB/s on MI300X) |
| **MFMA %** | Measured FLOPS / peak FP16 or FP8 MFMA throughput |

The kernel-level harness (`bench/harness.py`) reports the same plus
`arithmetic_intensity` and the roofline classification ("memory-bound" /
"compute-bound") via `slipstream.roofline`.

---

## Timing protocol

- **GPU events** (`torch.cuda.Event(enable_timing=True)`), not
  `time.perf_counter`. Sub-microsecond accuracy, no host-side noise.
- **Warmup:** 10 iterations discarded. Captures first-call JIT,
  hipBLASLt handle creation, Triton compile, autotune lookup.
- **Measurement:** 30 iterations.
- **Trim:** top and bottom 10% dropped before mean.
- **Synchronization:** `torch.cuda.synchronize()` between iterations.

---

## Workload grid (`--workload production`)

| Knob | Values |
|---|---|
| Model | `meta-llama/Meta-Llama-3-8B` |
| Batch size | 1, 8, 32, 128 |
| Prompt length | 512, 2048, 8192 |
| Decode steps | 128 (enough to amortize first-token cost) |
| dtype | fp16 |
| FP8 KV | {off, on} for each shape |

That's 4 × 3 × 2 = 24 workloads × N baselines.

`--workload smoke` is two workloads for quick development iteration.

---

## Baseline pin points

Each baseline is anchored to a specific upstream version so numbers stay
comparable across rebuilds:

| Baseline | Pin |
|---|---|
| vLLM-ROCm | `v0.6.5` (configurable via `SLIPSTREAM_VLLM_PIN`) |
| Flash-Attention-ROCm | latest ROCm wheel from upstream FA repo |
| hipBLASLt | bundled with the ROCm install (system) |
| torch.compile | `torch>=2.4` |

Each adapter records its observed version into `DecodeResult.metadata` so
results are auditable.

---

## Reproduction requirements

A number is "publishable" only when:

1. The exact `slipstream-bench` command and seed are documented.
2. The result is reproducible to within ±5% across three independent runs.
3. The autotune cache file in use is committed to the repo.
4. Baseline version pins are recorded.

Lossy speedup claims ("seems faster") never appear in the README or this doc.

---

## Result format

`results.json` is a list of `DecodeResult` dicts:

```json
[
  {
    "baseline": "slipstream",
    "input": {
      "model_id": "meta-llama/Meta-Llama-3-8B",
      "batch_size": 32, "prompt_len": 2048,
      "decode_steps": 128, "dtype": "fp16", "fp8_kv": true
    },
    "ms_per_token_p50": 4.32, "ms_per_token_p99": 4.91,
    "tokens_per_sec": 7407.4, "ttft_ms_p50": 38.2,
    "hbm_bw_gb_s": 4218.0, "measured_tflops": 162.4,
    "metadata": {"autotune_cache_hits": 31, "kv_blocks_used": 4096}
  },
  ...
]
```

---

## Comparing fairly

A few traps we explicitly avoid:

- **Different KV dtypes.** vLLM-ROCm at `kv_cache_dtype=auto` (fp16) vs.
  slipstream at FP8 KV is not an apples comparison. The grid always pairs
  matched configurations.
- **Different prompt distributions.** Both sides see the same synthetic
  prompts (controlled by `seed`).
- **First-call effects.** Warmup discards the first 10 iterations.
- **Free memory for the loser.** Each baseline is set up and torn down
  cleanly. Held HBM between runs would advantage whoever runs second.

---

## Where numbers will appear

When the headline numbers land, they go in **README.md → Headline results**
and a longer table in this file. Each row will link to the exact JSON dump
and command that produced it.
