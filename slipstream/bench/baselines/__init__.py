"""Adapters that wrap the baselines we benchmark against.

Each adapter implements :class:`BaselineProtocol` so the runner can treat
them uniformly. Imports are lazy — having vLLM / Flash-Attention installed
is not a hard dependency of slipstream itself; only the baselines you
actually exercise are required.

Registered baselines:

  - ``slipstream``         — our decode engine (this project)
  - ``vllm-rocm``          — vLLM upstream with the ROCm patches
  - ``flash-attention-rocm`` — Flash-Attention CK ``mha_decode`` directly
  - ``hipblaslt-fp16``     — raw hipBLASLt for the matmul comparisons
  - ``hipblaslt-fp8``      — raw hipBLASLt FP8 matmul
  - ``torch-eager``        — vanilla PyTorch ROCm
  - ``torch-compile``      — ``torch.compile(mode='max-autotune')``
"""

from slipstream.bench.baselines.protocol import (
    BaselineProtocol,
    BenchmarkInput,
    DecodeResult,
)

# Adapter implementations are imported lazily; we expose them as attribute
# accesses so a missing optional dependency doesn't break ``from
# slipstream.bench.baselines import *``.
__all__ = [
    "BaselineProtocol",
    "BenchmarkInput",
    "DecodeResult",
    "list_baselines",
    "make_baseline",
]


_BASELINE_FACTORIES = {
    "slipstream":            ("slipstream.bench.baselines.slipstream_adapter", "SlipstreamBaseline"),
    "vllm-rocm":             ("slipstream.bench.baselines.vllm_rocm",          "VLLMRocmBaseline"),
    "flash-attention-rocm":  ("slipstream.bench.baselines.fa_rocm",            "FlashAttentionRocmBaseline"),
    "hipblaslt-fp16":        ("slipstream.bench.baselines.hipblaslt",          "HipBLASLtFP16Baseline"),
    "hipblaslt-fp8":         ("slipstream.bench.baselines.hipblaslt",          "HipBLASLtFP8Baseline"),
    "torch-eager":           ("slipstream.bench.baselines.torch_baselines",    "TorchEagerBaseline"),
    "torch-compile":         ("slipstream.bench.baselines.torch_baselines",    "TorchCompileBaseline"),
}


def list_baselines() -> list[str]:
    return list(_BASELINE_FACTORIES.keys())


def make_baseline(name: str, **kwargs) -> BaselineProtocol:
    """Construct a baseline by name. Raises ``ImportError`` with an
    informative message if the backing package isn't installed.
    """
    if name not in _BASELINE_FACTORIES:
        raise KeyError(f"unknown baseline: {name!r}. Known: {list_baselines()}")
    mod_path, cls_name = _BASELINE_FACTORIES[name]
    import importlib
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise ImportError(
            f"Baseline {name!r} requires its backing package, which is not "
            f"installed. Underlying error: {e}"
        ) from e
    cls = getattr(mod, cls_name)
    return cls(**kwargs)
