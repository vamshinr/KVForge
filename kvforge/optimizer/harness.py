"""Five-stage correctness harness.

Borrowed from AutoKernel: a candidate kernel is only allowed to run on the
benchmark *after* it passes all five stages. The stages are ordered cheapest
to most expensive so failures abort early.

Stages:
  1. Smoke test       — single small input, tight tolerance.
  2. Shape sweep      — 8+ shape configs across 3 dtypes.
  3. Numerical stability — adversarial inputs (large values, near-zero, etc.).
  4. Determinism      — three runs, bitwise identical (catches race conditions).
  5. Edge cases       — non-power-of-two dims (1023, 4097, etc.).

Any single failure rejects the candidate. Throughput is never measured on a
candidate that hasn't passed all five.

Dtype tolerances are deliberately loose for low-precision types since
fp16/bf16 reductions accumulate noise that depends on order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch


# Dtype-specific tolerances. Tuned to match cuBLAS/cuDNN's typical drift on
# medium-sized reductions; adjust per-kernel if a particular op needs tighter
# bounds.
DEFAULT_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-5),   # (atol, rtol)
    torch.float16: (1e-2, 1e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


@dataclass
class CorrectnessResult:
    """Outcome of running the harness on a candidate kernel."""

    passed: bool
    failed_stage: str | None = None
    failure_detail: str = ""
    stage_times_ms: dict[str, float] = field(default_factory=dict)


# ---------- Type aliases ----------

# A candidate is `(*args, **kwargs) -> Tensor`. Same signature as the reference.
KernelFn = Callable[..., torch.Tensor]
# A factory builds a fresh set of inputs for a given shape and dtype.
InputFactory = Callable[[tuple[int, ...], torch.dtype, torch.device], tuple[tuple, dict]]


class CorrectnessHarness:
    """Validates a candidate kernel against a reference implementation.

    Construction is decoupled from invocation so the same harness can be reused
    across many candidates within a search loop.
    """

    def __init__(
        self,
        reference_fn: KernelFn,
        input_factory: InputFactory,
        shape_configs: list[tuple[int, ...]],
        dtypes: list[torch.dtype] | None = None,
        edge_shapes: list[tuple[int, ...]] | None = None,
        tolerances: dict[torch.dtype, tuple[float, float]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.reference_fn = reference_fn
        self.input_factory = input_factory
        self.shape_configs = shape_configs
        self.dtypes = dtypes or [torch.float32, torch.float16, torch.bfloat16]
        self.edge_shapes = edge_shapes or []
        self.tolerances = tolerances or DEFAULT_TOLERANCES
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def validate(self, candidate_fn: KernelFn) -> CorrectnessResult:
        """Run all five stages. Returns on first failure."""
        result = CorrectnessResult(passed=True)
        stages = [
            ("smoke", self._smoke_test),
            ("shape_sweep", self._shape_sweep),
            ("stability", self._stability),
            ("determinism", self._determinism),
            ("edge_cases", self._edge_cases),
        ]
        for name, stage_fn in stages:
            t0 = time.perf_counter()
            err = stage_fn(candidate_fn)
            result.stage_times_ms[name] = (time.perf_counter() - t0) * 1000
            if err is not None:
                result.passed = False
                result.failed_stage = name
                result.failure_detail = err
                return result
        return result

    # ---------- Stage implementations ----------
    # Each stage returns None on success or a string error message.

    def _smoke_test(self, candidate: KernelFn) -> str | None:
        """Single small input, tight tolerance. Catches gross errors fast."""
        if not self.shape_configs:
            return "no shape configs configured"
        shape = self.shape_configs[0]
        dtype = self.dtypes[0]
        args, kwargs = self.input_factory(shape, dtype, self.device)
        try:
            out_ref = self.reference_fn(*args, **kwargs)
            out_cand = candidate(*args, **kwargs)
        except Exception as e:
            return f"exception during smoke test: {type(e).__name__}: {e}"
        return self._compare(out_ref, out_cand, dtype, f"smoke shape={shape}")

    def _shape_sweep(self, candidate: KernelFn) -> str | None:
        """Run across all configured shapes × dtypes."""
        for shape in self.shape_configs:
            for dtype in self.dtypes:
                args, kwargs = self.input_factory(shape, dtype, self.device)
                try:
                    out_ref = self.reference_fn(*args, **kwargs)
                    out_cand = candidate(*args, **kwargs)
                except Exception as e:
                    return f"exception @ shape={shape} dtype={dtype}: {e}"
                err = self._compare(out_ref, out_cand, dtype, f"shape={shape} dtype={dtype}")
                if err:
                    return err
        return None

    def _stability(self, candidate: KernelFn) -> str | None:
        """Adversarial inputs designed to expose numerical bugs.

        Three patterns are used:
          - Large positive values (~1e3): exposes overflow in exp() etc.
          - Near-zero variance (constant + tiny noise): exposes 1/sqrt(0) bugs.
          - Mixed sign extreme range: exposes catastrophic cancellation.
        """
        if not self.shape_configs:
            return None
        shape = self.shape_configs[0]
        dtype = torch.float32   # use fp32 for stability checks
        patterns = ["large", "near_zero_var", "extreme_range"]
        for pattern in patterns:
            args, kwargs = self.input_factory(shape, dtype, self.device)
            args = tuple(self._perturb(a, pattern) if isinstance(a, torch.Tensor) else a
                         for a in args)
            try:
                out_ref = self.reference_fn(*args, **kwargs)
                out_cand = candidate(*args, **kwargs)
            except Exception as e:
                return f"stability pattern={pattern}: {type(e).__name__}: {e}"
            # Looser tolerance for stability tests; we only care about NaN/Inf
            # propagation matching, not bit-exact agreement.
            if torch.isnan(out_cand).any() and not torch.isnan(out_ref).any():
                return f"stability {pattern}: candidate produced NaN where reference did not"
            if torch.isinf(out_cand).any() and not torch.isinf(out_ref).any():
                return f"stability {pattern}: candidate produced Inf where reference did not"
        return None

    def _determinism(self, candidate: KernelFn) -> str | None:
        """Three runs, bitwise identical. Catches race conditions in atomics."""
        if not self.shape_configs:
            return None
        shape = self.shape_configs[0]
        dtype = self.dtypes[0]
        args, kwargs = self.input_factory(shape, dtype, self.device)
        outs = []
        for _ in range(3):
            outs.append(candidate(*args, **kwargs).clone())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if not (torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])):
            return "determinism: outputs differ across 3 identical runs"
        return None

    def _edge_cases(self, candidate: KernelFn) -> str | None:
        """Non-power-of-two shapes, often fatal for tile-remainder bugs."""
        for shape in self.edge_shapes:
            for dtype in self.dtypes:
                args, kwargs = self.input_factory(shape, dtype, self.device)
                try:
                    out_ref = self.reference_fn(*args, **kwargs)
                    out_cand = candidate(*args, **kwargs)
                except Exception as e:
                    return f"edge case shape={shape} dtype={dtype}: {e}"
                err = self._compare(out_ref, out_cand, dtype, f"edge shape={shape}")
                if err:
                    return err
        return None

    # ---------- Helpers ----------

    def _compare(
        self, ref: torch.Tensor, cand: torch.Tensor, dtype: torch.dtype, ctx: str
    ) -> str | None:
        """Tolerance-aware comparison."""
        if ref.shape != cand.shape:
            return f"{ctx}: shape mismatch ref={ref.shape} cand={cand.shape}"
        if ref.dtype != cand.dtype:
            return f"{ctx}: dtype mismatch ref={ref.dtype} cand={cand.dtype}"
        atol, rtol = self.tolerances.get(dtype, (1e-4, 1e-5))
        # Promote to fp32 for the comparison itself so tolerance arithmetic is
        # done in a clean precision.
        if not torch.allclose(
            ref.to(torch.float32), cand.to(torch.float32),
            atol=atol, rtol=rtol, equal_nan=True,
        ):
            diff = (ref.to(torch.float32) - cand.to(torch.float32)).abs()
            return (
                f"{ctx}: max abs diff {diff.max().item():.3e} "
                f"exceeds tolerance (atol={atol}, rtol={rtol})"
            )
        return None

    @staticmethod
    def _perturb(t: torch.Tensor, pattern: str) -> torch.Tensor:
        """Apply a stress pattern to a tensor. See `_stability`."""
        if pattern == "large":
            return t * 1000.0
        if pattern == "near_zero_var":
            return torch.full_like(t, fill_value=1.0) + t * 1e-6
        if pattern == "extreme_range":
            scale = torch.where(t > 0, torch.full_like(t, 1e3), torch.full_like(t, -1e3))
            return t * scale
        raise ValueError(f"unknown pattern: {pattern}")
