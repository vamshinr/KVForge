# Design Decisions

This document records the major design choices in slipstream and the
alternatives that were considered. It exists because the question reviewers
actually ask is rarely "what does the code do?" — it's "why did you do it
this way?"

---

## 1. System-level project, not a kernel-generation system

**Choice:** slipstream's contribution is the **decode-optimized system**
around the kernels — paged FP8 KV cache, continuous batcher, autotune
cache, and the integration that turns those into a runnable decode loop.
We do not generate kernels; we don't even hand-write specialized variants.

**Alternatives considered:**
- An LLM-driven kernel generator (Gimlet's kforge, Mirage, KernelBench-style).
- A library of hand-tuned Triton kernels per shape/dtype/GPU.
- An MLIR / custom-compiler approach.

**Reasoning:**
- Kernel-generation systems exist and are improving fast. Adding another
  smaller version of one would be derivative.
- Hand-tuning doesn't scale: one variant per shape per dtype per GPU
  generation. It's exactly the failure mode kernel generators are designed
  to replace.
- The **system** layer above kernels (paged KV with FP8, continuous
  batching, scheduler, engine glue) is where real production wins live on
  AMD today, and it's much less crowded than the kernel-codegen space.
- A system that consumes optimized kernels is **structurally complementary**
  to systems that generate them. The two compose.

**Tradeoff:** We're not the project to look at for "an LLM wrote a kernel
from scratch." We're the project to look at for "this decode path on
MI300X is faster end-to-end than vLLM-ROCm."

---

## 2. Parameterized Triton templates + autotune, not hand-tuned kernels

**Choice:** Each kernel we own (paged attention, FP8 GEMM) is a single
parameterized Triton template. All variation in `(BLOCK_M, BLOCK_N,
num_warps, num_stages, GROUP_M, SPLIT_K, ...)` lives in `@triton.autotune`
configs and is cached per `(shape_bucket, dtype, gfx_arch)` key.

**Alternatives considered:**
- Hand-tuned variants per shape (`paged_attn_b8_ctx2k_fp8.py`,
  `paged_attn_b32_ctx8k_fp16.py`, ...).
- `torch.compile` over the entire forward pass.
- Pure HIP / CK extensions.

**Reasoning:**
- Hand-tuning is a one-trick: tune *this* shape on *this* GPU. New shape =
  new variant. The maintenance cost grows linearly with shape coverage.
- Autotune is a one-time per-arch sweep. The result is a JSON cache file
  that's portable across machines with the same arch.
- Triton + ROCm now lowers `tl.dot` to `v_mfma_*` cleanly, so we're not
  giving up MFMA peak by staying in Triton.
- `torch.compile` is included as a *baseline* so we measure what Inductor's
  autotuner gives for free. It's not our hot path because Inductor doesn't
  know about paged KV or FP8 quantization.

**Tradeoff:** Triton's ROCm backend has known issues with some autotune
configs (`num_stages > 2` on gfx942 in certain templates). We pin
known-good configs and file upstream issues rather than reaching for HIP.

---

## 3. Reference implementation paired with every kernel

**Choice:** Every kernel ships paired with an eager PyTorch reference that
runs on CPU. The reference is the correctness oracle for tests and for
the autotune-time verification gate.

**Reasoning:**
- Triton kernels have a tight failure mode: a wrong tile-remainder mask
  produces silently corrupted outputs. A `torch.allclose` test against an
  obviously-correct reference catches this.
- The reference lets the entire test suite run on CPU dev machines (and CI).
  We don't need GPU access to test the scheduler, cache logic, FP8 quant,
  or attention math.
- The autotune sweep gates each candidate config on a correctness check
  against the reference before timing — protecting us from configurations
  that compile and run but produce wrong output.

**Tradeoff:** Maintaining two implementations is friction. Mitigation: the
reference is intentionally simple (slow, brute-force) and changes only
when the math changes.

---

## 4. vLLM-compatible KV layout (block_size = 16)

**Choice:** Block size is 16 by default, matching vLLM's convention.

**Reasoning:**
- During correctness work we can diff slipstream's KV cache state against
  vLLM's block-by-block. Any layout-related bug shows up immediately.
- vLLM picked 16 after substantial measurement — small blocks waste tail
  capacity on short sequences, large blocks waste compute on partial
  last-block decode. 16 balances both.
- Tooling that operates on KV cache dumps (debug viewers, ablation scripts)
  can be shared between projects.

**Tradeoff:** If a future workload favors a different block size, it's a
config knob — the layout machinery is parametric. Default stays at 16.

---

## 5. E4M3 for everything FP8

**Choice:** E4M3 (4-bit exponent, 3-bit mantissa) for KV cache, GEMM
activations, and GEMM weights.

**Alternatives considered:**
- E5M2 for KV (wider range, less precision).
- Mixed: E4M3 for activations, E5M2 for KV.

**Reasoning:**
- KV cache values are bounded by softmax outputs and per-token-scaled K
  activations. Range fits comfortably in E4M3 (±448 with appropriate scaling).
- Mantissa precision matters more than range for KV — we read those values
  back into attention scores, and a 3-bit-mantissa-equivalent error per
  element compounds across the full sequence.
- One format across the project means one set of scaling utilities, one
  test suite, one mental model.

**Tradeoff:** Long-context decode with extreme outlier tokens could in
principle benefit from E5M2 for the V cache. Per-token scaling mitigates
this in practice; if a real workload shows degradation we'll add a
constexpr switch.

---

## 6. Per-token, per-head KV scaling (not per-tensor)

**Choice:** FP8 KV cache uses **per-token, per-head** scales — one fp16
scale per `(token_position, head_index)` pair.

**Alternatives considered:**
- Per-tensor: one scale for the whole KV cache.
- Per-block: one scale per 16-token block.
- Per-element: a scale per dim (= no quantization, defeats the point).

**Reasoning:**
- Outlier tokens kill per-tensor and per-block scaling. A single token
  with `|x| > 100 * median` forces the scale up and crushes precision on
  every other token in the group.
- Per-token-per-head is the finest granularity that doesn't add overhead:
  one fp16 scale per group, < 1% of total KV bytes for typical head_dim.
- Quantization-aware fine-tuning literature converges on per-token scaling
  as the best precision-cost tradeoff.

---

## 7. Continuous batching, no preemption

**Choice:** The scheduler does continuous batching (mixed prefill/decode in
each step) but does not preempt in-flight sequences when the cache is full.
Pending requests wait.

**Alternatives considered:**
- vLLM-style KV swap to host RAM.
- Cooperative preemption (sequences yield voluntarily).
- Recompute-on-demand prefill (drop a sequence and re-run prefill if it
  comes back).

**Reasoning:**
- Swap-to-host is real engineering: PCIe transfer pacing, host buffer
  management, scheduling around the transfer. It's a Phase-5+ feature and
  warrants its own design doc.
- For decode workloads where requests are sized to fit the cache,
  preemption never fires. The skeleton's behavior is correct for the
  target workload.
- Documenting the limitation explicitly is better than hidden complexity.

---

## 8. Apache 2.0 license

**Choice:** Apache 2.0.

**Reasoning:** Explicit patent grant matters for GPU kernel code that sits
near vendor IP. De facto choice for serious infrastructure (PyTorch, vLLM,
SGLang, Triton). Permissive enough that the code can be lifted into
commercial products, which is the point.

---

## 9. No CI on GPU

**Choice:** CI runs CPU tests only. GPU-marked tests run manually before
release.

**Reasoning:** Self-hosted GPU runners cost real money. The CPU suite
covers the scheduler, cache, FP8 quant math, attention reference, GEMM
reference, autotune cache, and bench-harness logic — all the pure-Python
substrate. Kernel correctness on real hardware is verified before each
release; the surrounding framework changes more often and is what CI
guards.
