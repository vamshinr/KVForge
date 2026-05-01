# Contributing to KVForge

Thanks for your interest. KVForge is a research / portfolio project, but contributions are welcome — bug reports, kernel implementations, additional benchmarks, and documentation improvements are all useful.

## Development setup

```bash
git clone https://github.com/vamshinr/kvforge
cd kvforge
pip install -e ".[dev]"        # CPU dev environment
pip install -e ".[triton,dev]" # if you have a CUDA GPU
```

## Running tests

```bash
pytest tests/ -v                  # full suite
pytest tests/ -v -m "not gpu"     # CPU-only tests
pytest tests/test_kernels.py -v   # one test file
```

GPU-marked tests are auto-skipped when CUDA is unavailable.

## Code style

Run `ruff` before opening a PR:

```bash
ruff check kvforge/ tests/
ruff format kvforge/ tests/
```

The project uses Python 3.10+ type hints. Public APIs should have docstrings.

## Adding a new kernel

The pattern (modeled on `kvforge/kernels/rmsnorm.py`):

1. Create `kvforge/kernels/<name>.py` with three functions:
   - `<name>_reference(...)` — eager PyTorch implementation.
   - `<name>(...)` — public API. Falls back to reference if Triton is unavailable.
   - `<name>_bytes(shape, dtype)` and `<name>_flops(shape)` — roofline metadata.
2. Add it to `kvforge/kernels/__init__.py`.
3. Add tests in `tests/test_kernels.py` covering shape sweep, dtype sweep, edge cases, and determinism.
4. Add it to the bench CLI registry in `kvforge/bench/cli.py`.

The `CorrectnessHarness` in `kvforge/optimizer/harness.py` will validate your kernel automatically when invoked from the optimizer CLI.

## Reporting bugs

Include:
- Hardware (GPU model, driver version, CUDA version).
- PyTorch and Triton versions (`pip show torch triton`).
- Minimal reproducer.
- Expected vs actual output.

For correctness bugs, the reproducer should ideally be a failing test case in `tests/`.

## What's in scope

- New inference-relevant kernels (paged attention, fused linear+activation, INT8 GEMM).
- Better benchmark scripts and visualizations.
- Hardware support beyond CUDA (Apple Metal via MLX, AMD ROCm via HIP).
- LLM-driven candidate generation (the search loop is ready for it).

## What's out of scope

- Production inference serving (use vLLM or SGLang).
- Multi-GPU / distributed inference.
- Training kernels.
- Quantized model loading.

These are reasonable directions but a different project's problem.
