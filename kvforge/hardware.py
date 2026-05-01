"""Hardware detection and GPU spec database.

Auto-detects the active accelerator and looks up theoretical peak compute and memory
bandwidth so the roofline calculator and benchmark harness can normalize results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GPUSpec:
    """Hardware specifications for a single accelerator.

    Attributes
    ----------
    name: human-readable device name
    arch: shorthand architecture string ('hopper', 'ampere', 'ada', 'turing', 'apple', 'cpu')
    peak_fp16_tflops: theoretical FP16 dense throughput
    peak_fp32_tflops: theoretical FP32 dense throughput
    peak_bw_gb_s: peak HBM/DRAM bandwidth
    sm_count: streaming multiprocessor count (0 for non-CUDA)
    """

    name: str
    arch: str
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    peak_bw_gb_s: float
    sm_count: int = 0


# Specs sourced from public NVIDIA / AMD whitepapers. Conservative values
# (no sparsity, no tensor-core boost beyond standard FP16).
_GPU_DATABASE: dict[str, GPUSpec] = {
    "H100":     GPUSpec("NVIDIA H100",     "hopper",  989.5, 67.0, 3352.0, 132),
    "A100":     GPUSpec("NVIDIA A100",     "ampere",  312.0, 19.5, 2039.0, 108),
    "L4":       GPUSpec("NVIDIA L4",       "ada",     121.0, 30.3,  300.0,  58),
    "L40S":     GPUSpec("NVIDIA L40S",     "ada",     362.0, 91.6,  864.0, 142),
    "A10":      GPUSpec("NVIDIA A10",      "ampere",  125.0, 31.2,  600.0,  72),
    "T4":       GPUSpec("NVIDIA T4",       "turing",   65.0,  8.1,  300.0,  40),
    "RTX 4090": GPUSpec("NVIDIA RTX 4090", "ada",     330.0, 82.6, 1008.0, 128),
    "RTX 4080": GPUSpec("NVIDIA RTX 4080", "ada",     195.0, 48.7,  716.0,  76),
    "RTX 3090": GPUSpec("NVIDIA RTX 3090", "ampere",  142.0, 35.6,  936.0,  82),
    "RTX 3080": GPUSpec("NVIDIA RTX 3080", "ampere",  119.0, 29.8,  760.0,  68),
}


def detect_gpu() -> GPUSpec:
    """Return a GPUSpec for the currently active accelerator.

    Falls back to a generic CPU spec if no accelerator is available; an unknown CUDA
    device returns conservative estimates derived from `cudaGetDeviceProperties` so
    that benchmarks still produce roofline percentages.
    """
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GPUSpec("Apple Silicon", "apple", 10.0, 5.0, 200.0, 0)
        return GPUSpec("CPU", "cpu", 0.5, 0.5, 50.0, 0)

    device_name = torch.cuda.get_device_name(0)
    for key, spec in _GPU_DATABASE.items():
        if key in device_name:
            return spec

    # Unknown CUDA device — derive a conservative estimate.
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    # Rough heuristic: ~1 TFLOP per SM at modern clock speeds.
    estimated_tflops = sm_count * 1.0
    return GPUSpec(
        name=device_name,
        arch="unknown",
        peak_fp16_tflops=estimated_tflops * 2,
        peak_fp32_tflops=estimated_tflops,
        peak_bw_gb_s=500.0,  # conservative
        sm_count=sm_count,
    )


def has_triton() -> bool:
    """Return True if Triton is importable and a CUDA device is present."""
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
