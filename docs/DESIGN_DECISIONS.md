# Design Decisions

This document records the major design choices in KVForge and the alternatives that were considered. It exists because in interviews and code review, the question is rarely "what does the code do?" — it's "why did you do it this way?"

## 1. Triton over CUDA C++

**Choice:** All optimized kernels are written in Triton. CUDA C++ is not used.

**Alternatives considered:**
- CUDA C++ via `torch.utils.cpp_extension.load_inline` (AutoKernel's approach).
- A dual-backend system (AutoKernel ships both).
- Writing in pure Python and relying on `torch.compile` to lower to Triton.

**Reasoning:**
- Iteration speed dominates absolute peak performance for a search loop. Triton compiles in 1–5s; CUDA C++ takes 30+s. At 40 iterations/hour vs 5/hour, the agent loop converges 8× faster.
- For memory-bound kernels (RMSNorm, RoPE, softmax) — which is most of what KVForge optimizes — Triton routinely hits 80–95% of cuBLAS-equivalent throughput. The remaining gap doesn't justify the iteration penalty.
- Single backend keeps the codebase auditable. AutoKernel's dual-backend design is impressive but doubles the surface area for bugs.

**Tradeoff:** Compute-bound matmul kernels would benefit from CUDA C++ (direct WMMA control, register-level tuning). KVForge currently delegates matmul to cuBLAS rather than competing with it; a future v2 could add a CUDA C++ backend behind the same kernel interface.

## 2. Static candidate iterator vs LLM-driven generation

**Choice:** The search loop accepts a static `Iterator[(label, kernel_fn)]`. There is no LLM in the loop.

**Alternatives considered:**
- Wire up Anthropic / OpenAI APIs to generate candidates AutoKernel-style.
- Use an evolutionary algorithm (KernelFoundry's MAP-Elites approach).
- Use reinforcement learning (CUDA-L1's contrastive RL).

**Reasoning:**
- The agent loop's *structure* is what matters and what's interesting to demonstrate. The choice of candidate generator is orthogonal — swap in any of the above and the rest of the framework keeps working.
- Static iterators are deterministic and unit-testable. A test suite that depends on an external LLM API is brittle and expensive to run in CI.
- It separates concerns: this project shows the harness, scheduler, and verification logic. A real production system would plug an LLM into the same interface.

**Tradeoff:** The project doesn't show end-to-end "LLM writes kernel from scratch" capability. That's a different project (and the AutoKernel paper covers it well). Adding it later is a `~200 LOC` extension to the search loop.

## 3. Single-file kernel invariant

**Choice:** Each kernel lives in exactly one file (`kvforge/kernels/<name>.py`). No shared utility module for kernels.

**Alternatives considered:**
- Factor out common code (e.g., a `_next_power_of_two` helper) into `kvforge/kernels/common.py`.
- Use a class hierarchy with shared `Kernel` base class.

**Reasoning:**
- The agent loop's "edit one file, keep or revert" model breaks down if changes are coupled across files. Keeping each kernel self-contained means a candidate diff is always reviewable in isolation.
- Slight code duplication (the `_next_power_of_two` helper appears in three files) is acceptable and explicit. The alternative — a `common.py` that grows over time — is the kind of utility module that becomes a dumping ground.
- The pattern is borrowed from AutoKernel's program.md instructions, which enforce the same invariant via prompt.

**Tradeoff:** Code duplication is real. If we add 10 more kernels, the boilerplate adds up. The mitigation: factor out helpers when the duplication exceeds a threshold (~5 kernels, ~30 LOC duplicated each).

## 4. Roofline-guided tier selection

**Choice:** The roofline calculator classifies each kernel as compute-bound or memory-bound and recommends optimization tiers from a fixed playbook.

**Alternatives considered:**
- Let the search loop blindly try all tiers in order.
- Use a learned model to predict which tier will yield improvements.

**Reasoning:**
- The roofline classification is essentially free (~1µs per analysis) and provides a strong prior. Optimizing memory-bound kernels with compute-side techniques (tensor core utilization, accumulator precision) wastes search budget.
- The playbook is borrowed from AutoKernel and reflects real practitioner knowledge. Encoding it as data (not code) makes it inspectable.
- A learned model is overkill for a 6-tier playbook. The handful of cases where the heuristic is wrong (e.g., a kernel sitting near the ridge point) can be handled by trying both classifications.

## 5. Self-contained TinyLlama vs HuggingFace dependency

**Choice:** A self-contained TinyLlama-shaped decoder ships in `kvforge/models/tinyllama.py`. HuggingFace `transformers` is an optional extra (`pip install -e ".[hf]"`).

**Alternatives considered:**
- Require `transformers` for everything.
- Use `torch.hub` to download a pretrained checkpoint.

**Reasoning:**
- Recruiters and reviewers should be able to clone the repo, `pip install`, and run the smoke tests in 5 minutes without downloading multi-gigabyte checkpoints.
- The shape parameters (`hidden_size=2048`, `n_heads=32`, `n_kv_heads=4`, `head_dim=64`) match TinyLlama-1.1B's layer-by-layer profile. The kernel mix is identical.
- A "tiny" mode (`build_tinyllama(tiny=True)`) shrinks the model further (vocab=1k, hidden=256) for CPU smoke tests in CI.

**Tradeoff:** End-to-end inference benchmarks against real LLM outputs require the HF extra. This is documented in `BENCHMARKS.md`.

## 6. Five-stage correctness harness vs ad-hoc checks

**Choice:** Every candidate kernel must pass all five harness stages (smoke, shape sweep, stability, determinism, edge cases) before throughput is measured.

**Alternatives considered:**
- Single `assert torch.allclose(out, out_ref)` check.
- Continuous fuzzing during benchmark.

**Reasoning:**
- A 5× speedup on a kernel that produces wrong outputs is worse than useless. Silent numerical bugs in inference are devastating: they corrupt model output without any obvious failure signal. The harness explicitly catches each known bug class.
- Ordering matters: cheapest checks first means failed candidates abort in <1s instead of consuming the full benchmark window.
- The pattern is borrowed verbatim from AutoKernel — they describe in detail why each stage matters and the bug classes it catches.

**Tradeoff:** The harness adds ~30s to each iteration. For a 90-second iteration this is a 33% overhead that's unavoidable. There's no shortcut here — the bugs the harness catches are real and they happen.

## 7. Apache 2.0 license

**Choice:** Apache 2.0.

**Alternatives considered:** MIT, BSD-3, GPL.

**Reasoning:**
- Apache 2.0's explicit patent grant is important for a kernel optimization project. GPU kernel implementations sit close to vendor IP, and the patent grant protects users.
- It's the de facto license for serious infrastructure projects (PyTorch, TensorFlow, vLLM, SGLang, Triton itself).
- Permissive enough that the code can be lifted into commercial products, which is the point — the goal is to demonstrate engineering, not lock anyone in.

## 8. No CI on GPU

**Choice:** GitHub Actions runs the test suite on CPU only. GPU-marked tests are auto-skipped.

**Alternatives considered:**
- Self-hosted GPU runner.
- Free Colab runner via API.

**Reasoning:**
- Self-hosted GPU runners cost real money and require maintenance.
- The CI's purpose is to catch regressions in the pure-Python code (Amdahl ranker, classifier, harness logic, roofline math). All of that is covered by the CPU test suite.
- Kernel correctness on GPU is verified manually before each release. The kernels themselves rarely change once they pass the harness; the surrounding framework changes more often and is what CI guards.

**Tradeoff:** A regression in the Triton kernel itself wouldn't be caught by CI. Mitigation: the `pytest.mark.gpu` tests must be run manually before merge, and the harness ensures any kernel that ships has passed all five validation stages.
