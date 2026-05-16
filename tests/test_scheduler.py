"""Tests for the continuous-batching scheduler.

These run on CPU with a tiny mock cache; the scheduler is decoupled from
real KV math, so all of this is straight Python.
"""

from __future__ import annotations

import torch

from slipstream.kvcache.paged import PagedKVCache
from slipstream.scheduler.continuous import (
    FinishReason,
    Request,
    Scheduler,
)


def _make_cache(num_blocks: int = 32) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks, block_size=4,
        n_kv_heads=1, head_dim=8,
        kv_dtype=torch.float16, device="cpu",
    )


def _drive_one_token(scheduler: Scheduler, fake_next_token: int) -> int:
    """Run one engine step where every decision emits ``fake_next_token``.

    Returns the number of decisions processed (== total_tokens for decode-only
    steps, > num_decode_seqs when prefill chunks are present).
    """
    step = scheduler.step()
    n = len(step.decisions)
    for d in step.decisions:
        # The engine is the entity that actually appends K/V to the cache;
        # in this mock we just bump the seq length to keep block_table state
        # consistent with what the scheduler expects to see.
        k = torch.zeros(d.num_tokens, 1, 8, dtype=torch.float16)
        v = torch.zeros(d.num_tokens, 1, 8, dtype=torch.float16)
        scheduler.cache.append(d.request.seq_state, k, v)
        scheduler.commit_output(d, fake_next_token)
    return n


# ---------- Basic submit → prefill → decode flow ----------


def test_single_request_prefill_then_decode():
    cache = _make_cache()
    sched = Scheduler(cache, max_batched_tokens=512, chunk_size=512)

    req = Request(request_id=0, prompt_tokens=list(range(10)),
                  max_output_tokens=5, eos_token_id=None)
    sched.add(req)

    # Step 1: admit + prefill all 10 prompt tokens.
    step = sched.step()
    assert len(step.decisions) == 1
    assert step.decisions[0].is_prefill
    assert step.decisions[0].num_tokens == 10

    # Engine processes prefill; let's say it emits token 999 as the first output.
    d = step.decisions[0]
    cache.append(d.request.seq_state,
                 torch.zeros(d.num_tokens, 1, 8, dtype=torch.float16),
                 torch.zeros(d.num_tokens, 1, 8, dtype=torch.float16))
    sched.commit_output(d, next_token=999)
    assert req.output_tokens == [999]
    assert not req.is_prefilling

    # Step 2: pure decode of one token.
    step2 = sched.step()
    assert len(step2.decisions) == 1
    assert not step2.decisions[0].is_prefill
    assert step2.decisions[0].num_tokens == 1


def test_max_output_tokens_terminates_via_max_len():
    cache = _make_cache()
    sched = Scheduler(cache, max_batched_tokens=512, chunk_size=512)
    req = Request(request_id=0, prompt_tokens=[1, 2, 3],
                  max_output_tokens=3, eos_token_id=None)
    sched.add(req)

    # Prefill (emits 1st output) + 2 more decode steps = 3 outputs total.
    _drive_one_token(sched, fake_next_token=500)   # prefill, 1 output
    _drive_one_token(sched, fake_next_token=501)   # decode, 2 outputs
    _drive_one_token(sched, fake_next_token=502)   # decode, 3 outputs

    assert req.finished
    assert req.finish_reason == FinishReason.MAX_LEN
    assert len(req.output_tokens) == 3
    # Cache should be freed.
    assert sched.cache.block_table.num_free == cache.num_blocks


def test_eos_stops_generation_early():
    cache = _make_cache()
    sched = Scheduler(cache, chunk_size=512)
    req = Request(request_id=0, prompt_tokens=[1, 2],
                  max_output_tokens=100, eos_token_id=42)
    sched.add(req)

    _drive_one_token(sched, fake_next_token=42)   # prefill emits EOS immediately

    assert req.finished
    assert req.finish_reason == FinishReason.EOS
    assert req.output_tokens == [42]


# ---------- Chunked prefill ----------


def test_chunked_prefill_splits_long_prompts():
    cache = _make_cache(num_blocks=128)   # ≥ ceil(300/4) blocks for the prompt
    sched = Scheduler(cache, max_batched_tokens=512, chunk_size=128)

    long_prompt = list(range(300))
    req = Request(request_id=0, prompt_tokens=long_prompt,
                  max_output_tokens=2, eos_token_id=None)
    sched.add(req)

    # First chunk: 128 tokens, not yet finished prefilling.
    step1 = sched.step()
    assert step1.decisions[0].num_tokens == 128
    assert step1.decisions[0].is_prefill
    d = step1.decisions[0]
    cache.append(d.request.seq_state,
                 torch.zeros(128, 1, 8, dtype=torch.float16),
                 torch.zeros(128, 1, 8, dtype=torch.float16))
    # Engine returns no token until prefill is done.
    sched.commit_output(d, next_token=None if req.is_prefilling or req.prefill_offset + 128 < len(long_prompt) else 999)

    # The above passed next_token=None because req still has prompt remaining
    # after this chunk (128 < 300). Verify state.
    assert req.prefill_offset == 128

    # Second chunk: another 128.
    step2 = sched.step()
    assert step2.decisions[0].num_tokens == 128
    d = step2.decisions[0]
    cache.append(d.request.seq_state,
                 torch.zeros(128, 1, 8, dtype=torch.float16),
                 torch.zeros(128, 1, 8, dtype=torch.float16))
    sched.commit_output(d, next_token=None)
    assert req.prefill_offset == 256

    # Third chunk: 44 tokens, finishes prefill, emits first output.
    step3 = sched.step()
    assert step3.decisions[0].num_tokens == 44
    d = step3.decisions[0]
    cache.append(d.request.seq_state,
                 torch.zeros(44, 1, 8, dtype=torch.float16),
                 torch.zeros(44, 1, 8, dtype=torch.float16))
    sched.commit_output(d, next_token=777)
    assert req.prefill_offset == 300
    assert req.output_tokens == [777]


# ---------- Continuous batching: new requests join in-flight ----------


def test_new_request_joins_active_decode_step():
    cache = _make_cache(num_blocks=64)
    sched = Scheduler(cache, max_batched_tokens=64, chunk_size=64)

    r1 = Request(request_id=1, prompt_tokens=[1, 2, 3],
                 max_output_tokens=10, eos_token_id=None)
    sched.add(r1)
    _drive_one_token(sched, fake_next_token=100)   # r1 prefill + emit

    # Step 2: r1 is now decoding.
    # Add a new request while r1 is mid-stream.
    r2 = Request(request_id=2, prompt_tokens=[7, 8, 9, 10],
                 max_output_tokens=5, eos_token_id=None)
    sched.add(r2)

    step = sched.step()
    # Both should appear in the same step — r1 as a decode, r2 as a prefill.
    assert len(step.decisions) == 2
    kinds = sorted([(d.request.request_id, d.is_prefill) for d in step.decisions])
    assert kinds == [(1, False), (2, True)]
    assert step.num_decode_seqs == 1
    assert step.num_prefill_seqs == 1


# ---------- Backpressure: pending requests wait when cache is full ----------


def test_backpressure_blocks_admission_when_cache_full():
    # 2 blocks total, each holds 4 tokens → fits 8 prompt tokens.
    cache = _make_cache(num_blocks=2)
    sched = Scheduler(cache, chunk_size=512)

    # Keep `big` in flight so it holds both blocks.
    big = Request(request_id=0, prompt_tokens=list(range(8)),
                  max_output_tokens=100, eos_token_id=None)
    sched.add(big)
    _drive_one_token(sched, fake_next_token=100)   # admits + fills both blocks
    assert sched.cache.block_table.num_free == 0
    assert not big.finished

    # Now add another request — no free blocks available.
    blocked = Request(request_id=1, prompt_tokens=[1, 2, 3],
                      max_output_tokens=1, eos_token_id=None)
    sched.add(blocked)
    step = sched.step()
    assert blocked not in [d.request for d in step.decisions], \
        "blocked request should not be admitted while cache is full"
    assert sched.num_pending == 1
    # `big` should still be making decode progress.
    assert any(d.request is big and not d.is_prefill for d in step.decisions)
