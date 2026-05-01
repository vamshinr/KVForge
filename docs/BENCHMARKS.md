# Benchmarks: Methodology and Results

This document describes how KVForge's benchmarks are run, what they measure, and how to reproduce them on your own hardware.

## TL;DR

```bash
# Run the full per-kernel benchmark suite
python -m kvforge.bench --kernels rmsnorm,rope,softmax --dtype fp16 --iters 200

# Run a longer, more statistically stable run
python benchmarks/run_all.py --iters 500 --output benchmarks/results/

# Plot roofline visualization
python benchmarks/plot_roofline.py --input benchmarks/results/
```

## What we measure

Three baselines per kernel, three shape regimes per kernel, two dtypes (fp16 and bf16):

| Baseline | Code path |
|---|---|
| **Eager** | Pure PyTorch reference — `cuBLAS` for matmul, `ATen` decompositions for everything else |
| **`torch.compile`** | `torch.compile(fn, mode='max-autotune')` — TorchInductor generates and tunes its own Triton kernels |
| **KVForge** | The hand-tuned Triton kernel from `kvforge/kernels/` |

For each (kernel, baseline, shape, dtype) tuple we report:

1. **Wall-clock latency** in microseconds (CUDA event timing, trimmed mean of 200 runs)
2. **Speedup vs eager** and **speedup vs `torch.compile`**
3. **Achieved bandwidth** (GB/s) and **achieved throughput** (TFLOPS)
4. **Percent of roofline peak** at the kernel's arithmetic intensity

The percent-of-peak number is the most honest measurement. A 5× speedup vs eager that only hits 30% of bandwidth peak means there's still 3× left on the table; a 2× speedup that hits 85% of peak is essentially done.

## Timing methodology

GPU timing has three failure modes that destroy benchmark validity:

1. **CPU-side measurement**: `time.perf_counter()` measures CPU dispatch, not GPU execution.
2. **Async kernel launches**: a kernel submitted at T=0 may not start running until T=2µs and finish at T=10µs. Synchronizing at the wrong place gives garbage numbers.
3. **First-iteration overhead**: kernel JIT compilation, autotuning, and cuBLAS handle setup all hit the first call.

Our protocol addresses each:

```python
# Warmup outside timing window
for _ in range(warmup_iters):
    fn()
torch.cuda.synchronize()

# CUDA events bracket each iteration on-device
timings = []
for _ in range(bench_iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    timings.append(start.elapsed_time(end))

# Trimmed mean — drop top/bottom 10% for noise robustness
timings.sort()
trim = max(1, int(len(timings) * 0.1))
median = mean(timings[trim:-trim])
```

This is what the `BenchmarkHarness.time_fn()` method does. The 200-iteration default is enough to get sub-1% measurement noise on most kernels; raise to 500 or 1000 for kernels that show >2% variance.

## Correctness validation

Every result in the headline table comes from a kernel that passed all five stages of the correctness harness:

- **Smoke test** — single small input, tight tolerance.
- **Shape sweep** — 8+ shapes × 3 dtypes (24+ configs minimum).
- **Numerical stability** — adversarial inputs (large magnitudes, near-zero variance, extreme dynamic range).
- **Determinism** — 3 runs, bitwise identical outputs.
- **Edge cases** — non-power-of-two shapes (1023, 2047, 4097).

Run the validation explicitly:

```bash
pytest tests/test_kernels.py -v
pytest tests/test_harness.py -v
```

A failure in any of these is a bug. We don't report performance for kernels that don't pass.

## Hardware tested

The headline numbers in the README are from an NVIDIA L4 (24GB, 300 GB/s, 121 TF FP16). KVForge auto-detects the GPU via `kvforge.hardware.detect_gpu()` and looks up its specs from a built-in database covering H100, A100, L40S, L4, A10, T4, and RTX 3090/4090.

Variation across GPUs is significant for memory-bound kernels (most of what we benchmark). On bandwidth-rich H100 (3.3 TB/s), absolute speedups are smaller because the eager baseline is already closer to peak. On bandwidth-limited L4 (300 GB/s), the gap is wider.

## Reproducing the headline numbers

```bash
# Install with all extras
pip install -e ".[triton,dev]"

# 1. Profile a real model and see the kernel mix
python -m kvforge.profile --context 2048

# 2. Run the optimization loop (validates correctness, measures speedup)
python -m kvforge.optimize --kernels rmsnorm,rope,softmax

# 3. Full benchmark suite vs eager and torch.compile
python -m kvforge.bench --kernels rmsnorm,rope,softmax --iters 500

# 4. End-to-end model benchmark
python benchmarks/end_to_end.py --model tinyllama --context 2048
```

The numbers will differ from the README depending on your GPU. The relative speedups should be in the same ballpark — within ±20% on similar-class hardware.

## Known sources of variation

If you're seeing wildly different numbers from the README:

- **Power state.** GPUs throttle when warm. Check `nvidia-smi` clocks during the run; if they're below boost, run with `nvidia-smi -lgc <max>` to lock.
- **PCIe vs NVLink.** Multi-GPU systems may schedule the workload across the wrong device. Set `CUDA_VISIBLE_DEVICES=0` explicitly.
- **CUDA / Triton versions.** Triton 2.2 vs 3.0 produce different kernel code for the same source. Pin versions in the install.
- **Background load.** Other processes on the same GPU compete for SMs. Run on an idle device.

## Limitations

- **No multi-GPU benchmarks.** Single-device only.
- **No long-context benchmarks.** Context lengths beyond 4096 push KV cache off-device on smaller GPUs and the bottleneck shifts to PCIe.
- **No quantized baselines.** INT8 / INT4 / FP4 inference is increasingly common in production but isn't covered here.

These are all reasonable v2 directions but not in scope for the current release.
