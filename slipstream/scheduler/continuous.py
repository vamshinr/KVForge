"""Continuous-batching scheduler.

A request enters as :class:`Request` (prompt tokens + max output length).
The scheduler walks through ``step()``s:

  - For each active sequence currently in decode, schedule one token.
  - If there is remaining token-budget in the step, pull a pending request
    and schedule a prefill chunk (or, if the prompt fits, full prefill).
  - Each step is bounded by ``max_tokens_per_step`` so the engine has a
    predictable batch size.

The engine layer (``slipstream.engine``) is responsible for actually running
the kernel work; the scheduler only decides *what* to run.

Failure modes intentionally surfaced:

  - **KV cache full.** Pending requests stay pending. We do not preempt
    in-flight sequences in this skeleton — production systems would (vLLM
    swaps blocks to host RAM); we choose simplicity and document the
    behavior. Engine reports the backpressure to the caller.
  - **Sequence exceeds max_seq_len.** Marked finished with a flag.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from slipstream.kvcache.paged import PagedKVCache, SequenceState


class FinishReason(str, Enum):
    """Why a sequence stopped generating."""
    EOS = "eos"                # model emitted EOS token
    MAX_LEN = "max_len"        # hit max_output_tokens
    LEN_LIMIT = "len_limit"    # hit engine's max_seq_len
    USER_STOP = "user_stop"    # caller cancelled


@dataclass
class Request:
    """A request waiting to start or in flight."""

    request_id: int
    prompt_tokens: list[int]
    max_output_tokens: int
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    # Filled in by the scheduler as the request progresses.
    output_tokens: list[int] = field(default_factory=list)
    seq_state: SequenceState | None = None
    finished: bool = False
    finish_reason: FinishReason | None = None
    prefill_offset: int = 0    # how many prompt tokens already prefilled

    @property
    def prompt_remaining(self) -> int:
        return len(self.prompt_tokens) - self.prefill_offset

    @property
    def is_prefilling(self) -> bool:
        return self.prompt_remaining > 0

    @property
    def total_generated(self) -> int:
        return len(self.output_tokens)


@dataclass
class SchedulingDecision:
    """One sequence's contribution to the next engine step."""

    request: Request
    # tokens that this step will feed to the model for this sequence.
    # decode: 1 token (the last generated). prefill: up to chunk_size tokens.
    tokens: list[int]
    is_prefill: bool

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)


@dataclass
class SchedulerStep:
    """A fully-formed batch of work for the engine to execute."""

    decisions: list[SchedulingDecision]

    @property
    def total_tokens(self) -> int:
        return sum(d.num_tokens for d in self.decisions)

    @property
    def num_prefill_seqs(self) -> int:
        return sum(1 for d in self.decisions if d.is_prefill)

    @property
    def num_decode_seqs(self) -> int:
        return sum(1 for d in self.decisions if not d.is_prefill)

    def is_empty(self) -> bool:
        return len(self.decisions) == 0


class Scheduler:
    """Continuous-batching scheduler over a :class:`PagedKVCache`.

    Parameters
    ----------
    cache:
        Backing KV cache. The scheduler allocates/frees through it.
    max_seq_len:
        Hard cap on (prompt + output) tokens per sequence.
    max_batched_tokens:
        Cap on total tokens (decode + prefill chunks) per step. This is the
        budget that bounds kernel-launch latency.
    max_concurrent_seqs:
        Cap on simultaneously in-flight sequences. Mostly a safety knob —
        in practice cache capacity binds first.
    chunk_size:
        Max prefill tokens per request per step. Larger = better prefill
        throughput; smaller = more responsive to decode latency.
    """

    def __init__(
        self,
        cache: PagedKVCache,
        max_seq_len: int = 8192,
        max_batched_tokens: int = 4096,
        max_concurrent_seqs: int = 256,
        chunk_size: int = 512,
    ) -> None:
        self.cache = cache
        self.max_seq_len = max_seq_len
        self.max_batched_tokens = max_batched_tokens
        self.max_concurrent_seqs = max_concurrent_seqs
        self.chunk_size = chunk_size

        self._pending: deque[Request] = deque()
        self._active: list[Request] = []        # in-flight (prefill OR decode)
        self._next_seq_id = 0

    # ---------- Public API ----------

    def add(self, req: Request) -> None:
        """Enqueue a fresh request. Will start at the next step that has room."""
        if req.seq_state is not None:
            raise ValueError("request already submitted")
        self._pending.append(req)

    @property
    def num_active(self) -> int:
        return len(self._active)

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    def step(self) -> SchedulerStep:
        """Compose the next batch of work and return it.

        The engine should then:
          1. Run the kernels for ``step.decisions``.
          2. For each decision, call :meth:`commit_output` with the new K/V
             that came out of the layers and the token the LM head predicted.
        """
        decisions: list[SchedulingDecision] = []
        budget = self.max_batched_tokens

        # 1) Decode tokens for active sequences (latency-sensitive, packed first).
        for req in self._active:
            if req.finished:
                continue
            if req.is_prefilling:
                continue   # handled in pass 2
            if budget <= 0:
                break
            # The last token (output or last prompt) is the input for next decode.
            last_token = self._last_input_token(req)
            decisions.append(SchedulingDecision(
                request=req, tokens=[last_token], is_prefill=False,
            ))
            budget -= 1

        # 2) Prefill chunks for sequences currently mid-prefill.
        for req in self._active:
            if req.finished or budget <= 0:
                continue
            if req.is_prefilling:
                take = min(req.prompt_remaining, self.chunk_size, budget)
                if take <= 0:
                    continue
                chunk = req.prompt_tokens[req.prefill_offset:req.prefill_offset + take]
                decisions.append(SchedulingDecision(
                    request=req, tokens=chunk, is_prefill=True,
                ))
                budget -= take

        # 3) Admit new requests from the pending queue, subject to cache capacity.
        while (
            self._pending
            and budget > 0
            and len(self._active) < self.max_concurrent_seqs
        ):
            req = self._pending[0]
            # Only require enough free blocks for the first chunk — the rest
            # is allocated incrementally and naturally backpressures.
            need_blocks = self._blocks_needed_for_first_chunk(req)
            if need_blocks > self.cache.block_table.num_free:
                break   # backpressure: cache full
            # Allocate the sequence (no blocks yet — kvcache.append allocates lazily).
            req.seq_state = SequenceState(seq_id=self._allocate_seq_id())
            self._pending.popleft()
            self._active.append(req)

            take = min(req.prompt_remaining, self.chunk_size, budget)
            if take > 0:
                chunk = req.prompt_tokens[req.prefill_offset:req.prefill_offset + take]
                decisions.append(SchedulingDecision(
                    request=req, tokens=chunk, is_prefill=True,
                ))
                budget -= take

        return SchedulerStep(decisions=decisions)

    def commit_output(
        self,
        decision: SchedulingDecision,
        next_token: int | None,
    ) -> None:
        """Apply the engine's output for one decision.

        For prefill decisions, ``next_token`` is ignored except on the *final*
        chunk (when ``prefill_offset == len(prompt_tokens)``) — that token
        becomes the first output of the decode phase.

        The engine is responsible for having already appended the new K/V to
        the cache. The scheduler updates request bookkeeping only.
        """
        req = decision.request
        if decision.is_prefill:
            req.prefill_offset += decision.num_tokens
            assert req.prefill_offset <= len(req.prompt_tokens)
            if req.prefill_offset == len(req.prompt_tokens) and next_token is not None:
                # Transition to decode: the first generated token.
                req.output_tokens.append(next_token)
                self._maybe_finish(req, next_token)
        else:
            if next_token is not None:
                req.output_tokens.append(next_token)
                self._maybe_finish(req, next_token)
            else:
                # Engine failed to produce a token for this slot — finish gracefully.
                req.finished = True
                req.finish_reason = FinishReason.USER_STOP

        # Reap finished sequences (free their blocks once both we and engine
        # are done with the step).
        if req.finished and req.seq_state is not None:
            self.cache.free(req.seq_state)
            self._active = [r for r in self._active if not r.finished]

    # ---------- Internals ----------

    def _last_input_token(self, req: Request) -> int:
        """The most recently emitted token, used as the input for the next decode."""
        if req.output_tokens:
            return req.output_tokens[-1]
        # Decode-only path: edge case — sequence is in decode without any
        # generated token yet. This shouldn't happen if prefill emitted a
        # first token. Defensive fallback returns the last prompt token.
        return req.prompt_tokens[-1]

    def _blocks_needed_for_first_chunk(self, req: Request) -> int:
        """Blocks needed to make at least one productive prefill step.

        We do not require the whole prompt to fit before admitting — the
        engine will allocate blocks incrementally as more chunks land, and
        if the cache fills mid-prefill the request will simply stall until
        another sequence frees blocks. Admission gates on "enough room for
        the first chunk" so admission isn't impossibly conservative on long
        prompts.
        """
        first_chunk = min(len(req.prompt_tokens), self.chunk_size)
        return (first_chunk + self.cache.block_size - 1) // self.cache.block_size

    def _maybe_finish(self, req: Request, just_emitted: int) -> None:
        """Check stop conditions; mark finished and set reason if any fires."""
        if req.eos_token_id is not None and just_emitted == req.eos_token_id:
            req.finished, req.finish_reason = True, FinishReason.EOS
            return
        if just_emitted in req.stop_token_ids:
            req.finished, req.finish_reason = True, FinishReason.EOS
            return
        if req.total_generated >= req.max_output_tokens:
            req.finished, req.finish_reason = True, FinishReason.MAX_LEN
            return
        # Total length (prompt + generated) cap.
        total_len = len(req.prompt_tokens) + req.total_generated
        if total_len >= self.max_seq_len:
            req.finished, req.finish_reason = True, FinishReason.LEN_LIMIT

    def _allocate_seq_id(self) -> int:
        sid = self._next_seq_id
        self._next_seq_id += 1
        return sid
