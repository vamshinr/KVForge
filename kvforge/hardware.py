"""Hardware detection and GPU spec database.

Auto-detects the active accelerator and looks up theoretical peak compute and memory
bandwidth so the roofline calculator and benchmark harness can normalize results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUSpec:
    """Hardware specifications for a single accelerator.

    Attributes
    ----------
    name: human-readable device name
    arch: shorthand architecture string ('cdna2', 'cdna3', 'apple', 'cpu', ...)
    peak_fp16_tflops: theoretical FP16 matrix-core throughput (no sparsity)
    peak_fp32_tflops: theoretical FP32 vector throughput
    peak_bw_gb_s: peak HBM/DRAM bandwidth
    sm_count: compute unit count (0 for non-GPU)
    vendor: 'amd' | 'apple' | 'cpu'
    """

    name: str
    arch: str
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    peak_bw_gb_s: float
    sm_count: int = 0
    vendor: str = "amd"


# Specs sourced from public AMD whitepapers. Conservative values: matrix-core
# FP16 peak with FP32 accumulate, no sparsity; FP32 is vector-pipeline peak.
_GPU_DATABASE: dict[str, GPUSpec] = {
    "MI300X":   GPUSpec("AMD Instinct MI300X", "cdna3", 1307.4, 81.7, 5325.0, 304, "amd"),
    "MI300A":   GPUSpec("AMD Instinct MI300A", "cdna3",  980.6, 61.3, 5325.0, 228, "amd"),
    "MI250X":   GPUSpec("AMD Instinct MI250X", "cdna2",  383.0, 47.9, 3276.8, 220, "amd"),
    "MI250":    GPUSpec("AMD Instinct MI250",  "cdna2",  362.1, 45.3, 3276.8, 208, "amd"),
    "MI210":    GPUSpec("AMD Instinct MI210",  "cdna2",  181.0, 22.6, 1638.4, 104, "amd"),
}


def is_rocm() -> bool:
    """Return True if running on a ROCm (AMD) PyTorch build."""
    return getattr(torch.version, "hip", None) is not None


def detect_gpu() -> GPUSpec:
    """Return a GPUSpec for the currently active accelerator.

    Detects AMD Instinct GPUs on ROCm builds (``torch.cuda`` is the ROCm
    namespace), then Apple MPS, then CPU. Unknown GPUs fall back to a
    conservative estimate derived from device properties.
    """
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GPUSpec("Apple Silicon", "apple", 10.0, 5.0, 200.0, 0, "apple")
        return GPUSpec("CPU", "cpu", 0.5, 0.5, 50.0, 0, "cpu")

    device_name = torch.cuda.get_device_name(0)
    # AMD device names on ROCm sometimes appear as 'AMD Instinct MI300X' or
    # 'gfx942' depending on driver / PyTorch version; match both styles.
    name_upper = device_name.upper()
    for key, spec in _GPU_DATABASE.items():
        if key.upper() in name_upper:
            return spec
    # gfx ISA fallbacks for ROCm where the device name is the ISA only.
    gfx_map = {
        "GFX942": "MI300X",  # MI300X / MI300A both report gfx942; pick the X spec
        "GFX90A": "MI250X",
        "GFX908": "MI210",   # approximate — MI100 also gfx908
    }
    for gfx, key in gfx_map.items():
        if gfx in name_upper:
            return _GPU_DATABASE[key]

    # Unknown device — derive a conservative estimate.
    props = torch.cuda.get_device_properties(0)
    sm_count = getattr(props, "multi_processor_count", 0)
    estimated_tflops = sm_count * 1.0  # ~1 TFLOP per CU at modern clocks
    return GPUSpec(
        name=device_name,
        arch="rocm-unknown",
        peak_fp16_tflops=estimated_tflops * 2,
        peak_fp32_tflops=estimated_tflops,
        peak_bw_gb_s=500.0,
        sm_count=sm_count,
        vendor="amd",
    )


def has_triton() -> bool:
    """Return True if Triton is importable and a GPU device is present.

    On ROCm builds of PyTorch, ``torch.cuda.is_available()`` reports True for
    AMD devices; upstream Triton has supported AMD GPUs since 2.2.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
