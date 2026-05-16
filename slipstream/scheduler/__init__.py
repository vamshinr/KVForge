"""Continuous-batching scheduler.

The scheduler is the heart of the engine's throughput story. Each engine step
asks the scheduler for a batch of work; the scheduler picks decode tokens
(latency-sensitive) first and packs prefill chunks into the spare compute.

This is intentionally a small implementation — ~300 LOC, not a full vLLM
clone. It supports:

  - Continuous batching: new requests can join an in-flight step without
    waiting for the rest of the batch to finish.
  - Mixed prefill/decode in one step (the kernel kernels treat them as one
    contiguous query tensor with a per-token sequence-id list).
  - Chunked prefill: max ``chunk_size`` prompt tokens per step, so TTFT is
    bounded even for long contexts.
  - LRU eviction of completed sequences (KV blocks reclaimed at finish, not
    during).
"""

from slipstream.scheduler.continuous import (
    Request,
    Scheduler,
    SchedulerStep,
    SchedulingDecision,
)

__all__ = [
    "Request",
    "Scheduler",
    "SchedulerStep",
    "SchedulingDecision",
]
