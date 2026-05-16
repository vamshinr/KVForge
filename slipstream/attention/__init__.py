"""Decode attention.

The system-level contribution lives here: a single parameterized Triton
template for paged decode-step attention with grouped-query support and an
optional FP8 KV path. The template is autotuned (see ``slipstream.autotune``)
— there are no hand-tuned variants.

For correctness, every kernel ships paired with an eager PyTorch reference
in :mod:`slipstream.attention.reference`. The reference runs on CPU, knows
about the paged layout and FP8 KV, and is what the test suite measures
against.
"""

from slipstream.attention.reference import paged_attention_reference

__all__ = ["paged_attention_reference"]
