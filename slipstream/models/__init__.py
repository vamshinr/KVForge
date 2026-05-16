"""Self-contained model definitions for testing without external deps."""

from kvforge.models.tinyllama import build_tinyllama, make_forward_fn

__all__ = ["build_tinyllama", "make_forward_fn"]
