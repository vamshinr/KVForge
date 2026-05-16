"""Vanilla PyTorch and ``torch.compile`` engine baselines.

These two go through HuggingFace ``transformers`` for the model code, so
they're not the kernel SoTA — but they show what a typical PyTorch user
gets out-of-the-box, and ``torch.compile(mode="max-autotune")`` is a real
optimization target (Inductor often finds good Triton kernels on its own).

A successful slipstream story has to beat both with a wide margin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from slipstream.bench.baselines.protocol import BenchmarkInput, DecodeResult


@dataclass
class TorchEagerBaseline:
    name: str = "torch-eager"
    _model: object | None = None
    _tokenizer: object | None = None

    def setup(self, inp: BenchmarkInput) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for the torch-eager baseline. "
                "`pip install transformers accelerate`."
            ) from e

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[inp.dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(inp.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            inp.model_id, torch_dtype=dtype,
        ).to("cuda").eval()

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        import torch
        assert self._model is not None and self._tokenizer is not None

        # Synthetic ASCII prompts so we don't pull a dataset for benchmarks.
        prompts = ["word " * inp.prompt_len] * inp.batch_size
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        # Warmup.
        with torch.inference_mode():
            _ = self._model.generate(**enc, max_new_tokens=4, do_sample=False)
        torch.cuda.synchronize()

        with torch.inference_mode():
            t0 = time.perf_counter()
            out = self._model.generate(**enc, max_new_tokens=inp.decode_steps,
                                       do_sample=False)
            torch.cuda.synchronize()
            wall_s = time.perf_counter() - t0

        total_tokens = inp.batch_size * inp.decode_steps
        ms_per_token = (wall_s * 1000) / max(total_tokens, 1)
        return DecodeResult(
            baseline=self.name, input=inp,
            ms_per_token_p50=ms_per_token, ms_per_token_p99=ms_per_token,
            tokens_per_sec=total_tokens / max(wall_s, 1e-6),
            ttft_ms_p50=0.0,
            metadata={},
        )

    def teardown(self) -> None:
        self._model = None
        self._tokenizer = None
        import gc; gc.collect()


@dataclass
class TorchCompileBaseline:
    name: str = "torch-compile"
    mode: str = "max-autotune"
    _model: object | None = None
    _tokenizer: object | None = None

    def setup(self, inp: BenchmarkInput) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for the torch-compile baseline. "
                "`pip install transformers accelerate`."
            ) from e

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[inp.dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(inp.model_id)
        m = AutoModelForCausalLM.from_pretrained(
            inp.model_id, torch_dtype=dtype,
        ).to("cuda").eval()
        # Compiling the whole model is too aggressive for HF's generate loop;
        # we compile only the forward (Inductor's recommended mode for decode).
        m.forward = torch.compile(m.forward, mode=self.mode, fullgraph=False)
        self._model = m

    def run_workload(self, inp: BenchmarkInput) -> DecodeResult:
        # The mechanics are the same as eager; the model.forward is compiled.
        return TorchEagerBaseline.run_workload(self, inp)   # type: ignore[arg-type]

    def teardown(self) -> None:
        TorchEagerBaseline.teardown(self)   # type: ignore[arg-type]
