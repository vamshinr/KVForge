"""hipBLASLt baselines — for the GEMM side of the comparison.

These are *kernel* baselines, not engine baselines: they measure raw matmul
throughput against the parameterized slipstream FP8 GEMM template. The
shapes we sweep are the decode-skinny ones (small M, hidden N=K). hipBLASLt
is the SoTA AMD path for FP8 matmul today.

We invoke via ``torch.matmul`` on the appropriate fp8 tensors — PyTorch's
ROCm build routes FP8 matmul to hipBLASLt when the tensors are FP8 with
provided scales.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult


@dataclass
class HipBLASLtFP16Baseline:
    name: str = "hipblaslt-fp16"

    def setup(self, inp: BenchmarkInput) -> None:
        import torch
        if not torch.cuda.is_available():
            raise ImportError("hipBLASLt baseline requires a ROCm-enabled torch.cuda")

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        import torch
        M = inp.batch_size
        N = K = 4096
        device = torch.device("cuda")
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        for _ in range(10):
            _ = a @ b
        torch.cuda.synchronize()

        timings_ms: list[float] = []
        for _ in range(max(inp.decode_steps, 50)):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = a @ b
            e.record()
            e.synchronize()
            timings_ms.append(s.elapsed_time(e))

        timings_ms.sort()
        p50 = timings_ms[len(timings_ms) // 2]
        p99 = timings_ms[int(len(timings_ms) * 0.99)]
        return DecodeResult(
            baseline=self.name, input=inp,
            ms_per_token_p50=p50, ms_per_token_p99=p99,
            tokens_per_sec=M * 1000.0 / max(p50, 1e-6),
            ttft_ms_p50=0.0,
            metadata={"M": M, "N": N, "K": K, "kernel_only": True},
        )

    def teardown(self) -> None:
        pass


@dataclass
class HipBLASLtFP8Baseline:
    name: str = "hipblaslt-fp8"

    def setup(self, inp: BenchmarkInput) -> None:
        import torch
        if not torch.cuda.is_available():
            raise ImportError("hipBLASLt FP8 baseline requires a ROCm-enabled torch.cuda")
        # Probe whether the installed torch has FP8 matmul wired to hipBLASLt.
        # On older ROCm wheels this raises NotImplementedError at call time;
        # we let it fail naturally there.

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        import torch
        from slipstream.gemm.reference import (
            quantize_activation_per_token,
            quantize_weight_per_channel,
        )

        M = inp.batch_size
        N = K = 4096
        device = torch.device("cuda")
        a = torch.randn(M, K, dtype=torch.float16, device=device)
        w = torch.randn(K, N, dtype=torch.float16, device=device) * 0.02
        a_fp8, sa = quantize_activation_per_token(a)
        w_fp8, sb = quantize_weight_per_channel(w)

        def _run():
            # torch._scaled_mm is the public FP8 matmul entrypoint in 2.4+.
            # Signature: (a, b, scale_a, scale_b, ... ) -> tensor in out_dtype.
            return torch._scaled_mm(
                a_fp8, w_fp8,
                scale_a=sa.float().reshape(-1, 1),
                scale_b=sb.float().reshape(1, -1),
                out_dtype=torch.float16,
            )

        for _ in range(10):
            _ = _run()
        torch.cuda.synchronize()

        timings_ms: list[float] = []
        for _ in range(max(inp.decode_steps, 50)):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = _run()
            e.record()
            e.synchronize()
            timings_ms.append(s.elapsed_time(e))

        timings_ms.sort()
        p50 = timings_ms[len(timings_ms) // 2]
        p99 = timings_ms[int(len(timings_ms) * 0.99)]
        return DecodeResult(
            baseline=self.name, input=inp,
            ms_per_token_p50=p50, ms_per_token_p99=p99,
            tokens_per_sec=M * 1000.0 / max(p50, 1e-6),
            ttft_ms_p50=0.0,
            metadata={"M": M, "N": N, "K": K, "kernel_only": True},
        )

    def teardown(self) -> None:
        pass
