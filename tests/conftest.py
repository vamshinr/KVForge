"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU-marked tests when CUDA is unavailable."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="requires CUDA GPU")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
