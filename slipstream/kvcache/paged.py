"""Paged KV cache with block-table indirection.

The cache stores a global pool of fixed-size blocks. Each sequence holds an
ordered list of physical block ids (its *block table*); appending a new token
either fills the current tail block or allocates a fresh one from the free
list. Blocks are never moved or reshuffled — only appended, freed at sequence
end, or reused via the free list. This is the layout vLLM uses, and it lets
the attention kernel walk through KV one block at a time with a single
indirection per block.

Block size = 16 is the default. Smaller block sizes waste tail capacity on
short sequences; larger block sizes waste compute on the last (partial)
block during decode. 16 is the sweet spot vLLM landed on after measurement,
so we adopt it to make block-by-block diffing against vLLM feasible.

FP8 KV is selected at construction time via ``kv_dtype``. When FP8, the cache
additionally stores per-(token, head) scales in FP16. The attention kernel is
responsible for the dequantize step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SequenceState:
    """Live state for one in-flight sequence."""

    seq_id: int
    block_ids: list[int] = field(default_factory=list)
    length: int = 0   # tokens currently stored, including all prefill

    def num_blocks_needed(self, block_size: int) -> int:
        return (self.length + block_size - 1) // block_size

    def slot_for(self, position: int, block_size: int) -> tuple[int, int]:
        """Return (physical_block_id, slot_within_block) for a token position.

        ``position`` is the absolute token index from the start of the
        sequence. Raises ``IndexError`` if the position is past the current
        length.
        """
        if position < 0 or position >= self.length:
            raise IndexError(
                f"position {position} out of range for sequence of length {self.length}"
            )
        logical_block = position // block_size
        slot = position % block_size
        return self.block_ids[logical_block], slot


class BlockTable:
    """Free-list allocator over a fixed pool of physical block ids.

    Concurrency: not thread-safe. The scheduler is single-threaded by design;
    if multi-threaded scheduling is ever added, a lock goes around alloc/free.
    """

    def __init__(self, num_blocks: int) -> None:
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # LIFO
        self._num_blocks = num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(
                f"out of KV cache blocks: requested {n}, free {len(self._free)}"
            )
        out = [self._free.pop() for _ in range(n)]
        return out

    def free(self, block_ids: list[int]) -> None:
        # LIFO improves cache locality of reuse (the just-freed block is
        # hot in L2 from the sequence that just released it).
        for b in reversed(block_ids):
            self._free.append(b)


class PagedKVCache:
    """Global KV cache backed by a pool of fixed-size blocks.

    Layout per block (PyTorch tensors live on ``device``):

      ``k`` shape: ``[num_blocks, block_size, n_kv_heads, head_dim]``
      ``v`` shape: same as ``k``
      (optional, FP8 only)
      ``k_scales`` shape: ``[num_blocks, block_size, n_kv_heads]``  fp16
      ``v_scales`` shape: same as ``k_scales``

    The leading ``num_blocks`` axis is indexed via the per-sequence block
    table. Within a block, tokens are dense along the ``block_size`` axis.

    Parameters
    ----------
    num_blocks: pool capacity. Sized at engine startup by
        ``available_hbm / per_block_bytes``.
    block_size: tokens per block. 16 is the default.
    n_kv_heads: number of K/V heads (after GQA grouping; equal to query head
        count divided by the GQA group size).
    head_dim: per-head dimension.
    kv_dtype: storage dtype for K and V. ``torch.float16`` or
        ``torch.float8_e4m3fn``.
    device: where the tensors live.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        kv_dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)

        shape = (num_blocks, block_size, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=kv_dtype, device=self.device)
        self.v = torch.zeros(shape, dtype=kv_dtype, device=self.device)

        self._is_fp8 = kv_dtype == torch.float8_e4m3fn
        if self._is_fp8:
            scale_shape = (num_blocks, block_size, n_kv_heads)
            self.k_scales = torch.ones(scale_shape, dtype=torch.float16, device=self.device)
            self.v_scales = torch.ones(scale_shape, dtype=torch.float16, device=self.device)
        else:
            self.k_scales = None
            self.v_scales = None

        self.block_table = BlockTable(num_blocks)

    @property
    def is_fp8(self) -> bool:
        return self._is_fp8

    # ---------- Sequence lifecycle ----------

    def append(
        self,
        seq: SequenceState,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        """Append ``n`` new tokens of K and V to a sequence.

        ``k_new`` and ``v_new`` shape: ``[n, n_kv_heads, head_dim]`` in fp16
        or bf16 (always the high-precision producer dtype — quantization to
        FP8 happens here if the cache is FP8).

        Allocates new blocks as needed; never moves existing blocks.
        """
        n = k_new.shape[0]
        assert k_new.shape == v_new.shape == (n, self.n_kv_heads, self.head_dim)

        for i in range(n):
            position = seq.length
            logical_block = position // self.block_size

            # Allocate a new physical block when crossing the tail boundary.
            if logical_block >= len(seq.block_ids):
                seq.block_ids.extend(self.block_table.allocate(1))

            pb = seq.block_ids[logical_block]
            slot = position % self.block_size

            self._write_slot(pb, slot, k_new[i], v_new[i])
            seq.length += 1

    def free(self, seq: SequenceState) -> None:
        """Return all of a sequence's blocks to the free list."""
        if seq.block_ids:
            self.block_table.free(seq.block_ids)
            seq.block_ids.clear()
        seq.length = 0

    # ---------- Reads ----------

    def gather_sequence(
        self,
        seq: SequenceState,
        out_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize a sequence's K and V into contiguous tensors.

        Shape: ``[seq.length, n_kv_heads, head_dim]`` in ``out_dtype``.

        This dequantizes FP8 if needed. Used by the *reference* attention
        path; real Triton kernels read blockwise without materialization.
        """
        if seq.length == 0:
            empty = torch.zeros(
                (0, self.n_kv_heads, self.head_dim),
                dtype=out_dtype, device=self.device,
            )
            return empty, empty.clone()

        n_full = seq.length // self.block_size
        n_tail = seq.length % self.block_size

        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []

        for i, pb in enumerate(seq.block_ids):
            if i < n_full:
                slots = slice(0, self.block_size)
            elif i == n_full and n_tail > 0:
                slots = slice(0, n_tail)
            else:
                break
            k_parts.append(self._read_slots(pb, slots, "k", out_dtype))
            v_parts.append(self._read_slots(pb, slots, "v", out_dtype))

        k = torch.cat(k_parts, dim=0)
        v = torch.cat(v_parts, dim=0)
        return k, v

    # ---------- Internals ----------

    def _write_slot(
        self,
        physical_block: int,
        slot: int,
        k_vec: torch.Tensor,    # [n_kv_heads, head_dim]
        v_vec: torch.Tensor,    # [n_kv_heads, head_dim]
    ) -> None:
        if self._is_fp8:
            from slipstream.kvcache.fp8 import quantize_fp8
            k_q, k_s = quantize_fp8(k_vec, group_dim=-1)
            v_q, v_s = quantize_fp8(v_vec, group_dim=-1)
            self.k[physical_block, slot] = k_q
            self.v[physical_block, slot] = v_q
            self.k_scales[physical_block, slot] = k_s.to(torch.float16)
            self.v_scales[physical_block, slot] = v_s.to(torch.float16)
        else:
            self.k[physical_block, slot] = k_vec.to(self.kv_dtype)
            self.v[physical_block, slot] = v_vec.to(self.kv_dtype)

    def _read_slots(
        self,
        physical_block: int,
        slots: slice,
        which: str,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        # which ∈ {"k", "v"}
        raw = (self.k if which == "k" else self.v)[physical_block, slots]
        if self._is_fp8:
            from slipstream.kvcache.fp8 import dequantize_fp8
            scales = (self.k_scales if which == "k" else self.v_scales)[physical_block, slots]
            return dequantize_fp8(raw, scales, out_dtype=out_dtype, group_dim=-1)
        return raw.to(out_dtype)

    # ---------- Capacity & utilization ----------

    def utilization(self) -> float:
        used = self.num_blocks - self.block_table.num_free
        return used / self.num_blocks if self.num_blocks else 0.0

    def per_block_bytes(self) -> int:
        """Bytes consumed per physical block, including scales for FP8."""
        elem_bytes = torch.tensor([], dtype=self.kv_dtype).element_size()
        # k + v
        b = 2 * self.block_size * self.n_kv_heads * self.head_dim * elem_bytes
        if self._is_fp8:
            # k_scales + v_scales (fp16)
            b += 2 * self.block_size * self.n_kv_heads * 2
        return b
