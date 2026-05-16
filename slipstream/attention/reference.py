"""Eager paged decode-attention with GQA and optional FP8 KV.

This is the correctness oracle for every paged-attention variant in
slipstream. It does the math the slow way — full softmax in fp32, no
streaming, no tile-level reductions — so it cannot be the production path,
but any kernel that disagrees with it on a (shape, seed) is wrong.

Math (per query token in a decode step):

    For each query head h_q (n_q_heads total):
        h_kv = h_q // group_size                 # GQA mapping
        scores = q[h_q] @ K[h_kv].T / sqrt(d)    # [seq_len]
        probs  = softmax(scores)                  # [seq_len], fp32
        out[h_q] = probs @ V[h_kv]                # [head_dim]

The "paged" part is just an indirection: ``K`` and ``V`` come from the
:class:`PagedKVCache` rather than contiguous tensors. The reference
materializes the contiguous view via ``gather_sequence`` and does the math
on it. Production kernels skip the materialization and walk blocks directly,
which is why this is reference-only.
"""

from __future__ import annotations

import math

import torch

from slipstream.kvcache.paged import PagedKVCache, SequenceState


def paged_attention_reference(
    q: torch.Tensor,
    cache: PagedKVCache,
    seqs: list[SequenceState],
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode-step paged attention. One query token per sequence.

    Parameters
    ----------
    q:
        Query tensor for the current decode step.
        Shape: ``[batch, n_q_heads, head_dim]`` in fp16 / bf16.
    cache:
        The paged KV cache holding all prior K/V for these sequences.
    seqs:
        One :class:`SequenceState` per batch element. ``len(seqs) == batch``
        and ``seqs[i].length`` is the number of context tokens (the K/V
        positions to attend over) — the just-appended query is *not* part of
        this length; append it after attention.
    scale:
        Softmax scale. Defaults to ``1 / sqrt(head_dim)``.
    compute_dtype:
        Dtype for scores and softmax. fp32 is the only sane choice for the
        reference; kernels may use lower precision internally with their own
        tolerances.

    Returns
    -------
    Output tensor with shape ``[batch, n_q_heads, head_dim]`` in ``q.dtype``.

    Constraints
    -----------
    GQA group size is inferred from ``n_q_heads / cache.n_kv_heads`` and must
    be an integer. The function asserts this.
    """
    batch, n_q_heads, head_dim = q.shape
    assert head_dim == cache.head_dim, (
        f"head_dim mismatch: q={head_dim} cache={cache.head_dim}"
    )
    assert n_q_heads % cache.n_kv_heads == 0, (
        f"n_q_heads ({n_q_heads}) not divisible by n_kv_heads ({cache.n_kv_heads}) — "
        "GQA group size must be an integer"
    )
    assert len(seqs) == batch, f"batch mismatch: q[0]={batch} seqs={len(seqs)}"

    group_size = n_q_heads // cache.n_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.zeros_like(q)

    for b, seq in enumerate(seqs):
        if seq.length == 0:
            # No context yet — output is undefined in practice but zero is a
            # benign placeholder. Real serving never gets here.
            continue

        # K_seq, V_seq: [seq_len, n_kv_heads, head_dim] in compute_dtype-ish.
        k_seq, v_seq = cache.gather_sequence(seq, out_dtype=compute_dtype)
        q_b = q[b].to(compute_dtype)                # [n_q_heads, head_dim]

        # Expand K/V along the head axis to match Q via GQA broadcasting.
        # k_expanded: [seq_len, n_q_heads, head_dim]
        k_expanded = k_seq.repeat_interleave(group_size, dim=1)
        v_expanded = v_seq.repeat_interleave(group_size, dim=1)

        # scores[h, t] = q_b[h] · K[t, h]   shape: [n_q_heads, seq_len]
        scores = torch.einsum("hd,thd->ht", q_b, k_expanded) * scale
        probs = torch.softmax(scores, dim=-1)        # [n_q_heads, seq_len]

        # out_b[h, d] = sum_t probs[h, t] * V[t, h, d]
        out_b = torch.einsum("ht,thd->hd", probs, v_expanded)
        out[b] = out_b.to(q.dtype)

    return out


def paged_attention_reference_prefill(
    q: torch.Tensor,
    cache: PagedKVCache,
    seqs: list[SequenceState],
    q_lens: list[int],
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Prefill variant: many query tokens per sequence, with causal masking.

    Parameters
    ----------
    q:
        Concatenated query for all prefill tokens across the batch.
        Shape: ``[sum(q_lens), n_q_heads, head_dim]``.
    seqs:
        Sequence states *after* the prefill K/V have already been appended.
        ``seqs[b].length`` therefore equals the post-prefill context length.
    q_lens:
        Number of query tokens belonging to each batch element. Their absolute
        positions are the last ``q_lens[b]`` of the sequence.

    Notes
    -----
    Prefill exists in the reference because the engine needs an end-to-end
    correctness baseline. Production prefill goes through Flash-Attention-ROCm
    on real hardware (well-served by upstream).
    """
    total_q, n_q_heads, head_dim = q.shape
    assert sum(q_lens) == total_q, f"q_lens {q_lens} sum to {sum(q_lens)} ≠ {total_q}"
    assert n_q_heads % cache.n_kv_heads == 0
    group_size = n_q_heads // cache.n_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    q_offset = 0

    for b, (seq, ql) in enumerate(zip(seqs, q_lens, strict=True)):
        if ql == 0:
            continue
        # The ql query tokens are the most recent ql positions of the sequence.
        q_start_pos = seq.length - ql

        k_seq, v_seq = cache.gather_sequence(seq, out_dtype=compute_dtype)
        q_chunk = q[q_offset:q_offset + ql].to(compute_dtype)    # [ql, n_q_heads, d]

        k_expanded = k_seq.repeat_interleave(group_size, dim=1)
        v_expanded = v_seq.repeat_interleave(group_size, dim=1)

        # scores[q_idx, h, k_idx]
        scores = torch.einsum("qhd,khd->qhk", q_chunk, k_expanded) * scale

        # Causal mask: query at position (q_start_pos + qi) can attend to
        # keys at positions [0, q_start_pos + qi]. Anything beyond is -inf.
        seq_len = seq.length
        q_positions = torch.arange(q_start_pos, q_start_pos + ql, device=q.device)
        k_positions = torch.arange(seq_len, device=q.device)
        causal_mask = k_positions[None, :] > q_positions[:, None]   # [ql, seq_len]
        scores = scores.masked_fill(causal_mask[:, None, :], float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        out_chunk = torch.einsum("qhk,khd->qhd", probs, v_expanded)

        out[q_offset:q_offset + ql] = out_chunk.to(q.dtype)
        q_offset += ql

    return out
