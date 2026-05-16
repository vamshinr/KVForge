"""CPU-runnable smoke demo of slipstream's foundation.

Exercises the paged KV cache + continuous-batching scheduler with mock
"kernels" (just zero tensors and placeholder tokens). Demonstrates the
mixed prefill/decode flow, chunked prefill, EOS-driven termination, and
KV-block reclamation.

Run::

    python examples/scheduler_demo.py
"""

from __future__ import annotations

import torch

from slipstream.kvcache.paged import PagedKVCache
from slipstream.scheduler.continuous import Request, Scheduler


def main() -> int:
    print("== slipstream scheduler smoke demo ==")
    cache = PagedKVCache(
        num_blocks=256, block_size=16,
        n_kv_heads=2, head_dim=64,
        kv_dtype=torch.float16, device="cpu",
    )
    sched = Scheduler(
        cache=cache, max_seq_len=1024,
        max_batched_tokens=512, max_concurrent_seqs=8, chunk_size=128,
    )

    # Three requests with different shapes.
    sched.add(Request(request_id=1, prompt_tokens=list(range(50)),
                      max_output_tokens=10, eos_token_id=42))
    sched.add(Request(request_id=2, prompt_tokens=list(range(300)),
                      max_output_tokens=5, eos_token_id=None))
    sched.add(Request(request_id=3, prompt_tokens=list(range(20)),
                      max_output_tokens=8, eos_token_id=99))

    step_idx = 0
    while sched.num_active or sched.num_pending:
        step = sched.step()
        if step.is_empty():
            break
        print(f"step {step_idx:2d}: {len(step.decisions)} decisions   "
              f"prefill={step.num_prefill_seqs}  decode={step.num_decode_seqs}  "
              f"tokens={step.total_tokens}  "
              f"kv_free={cache.block_table.num_free}/{cache.num_blocks}")
        for d in step.decisions:
            shape = (d.num_tokens, cache.n_kv_heads, cache.head_dim)
            k = torch.zeros(shape, dtype=torch.float16)
            v = torch.zeros(shape, dtype=torch.float16)
            cache.append(d.request.seq_state, k, v)
            # Emit a fake token only when prefill is complete or we're decoding.
            fake_tok = None
            if not d.is_prefill or d.request.prompt_remaining == 0:
                # Make req 1 emit EOS after 3 outputs so we test early termination.
                if d.request.request_id == 1 and len(d.request.output_tokens) >= 2:
                    fake_tok = 42
                else:
                    fake_tok = 100 + d.request.request_id
            sched.commit_output(d, fake_tok)
        step_idx += 1

    print(f"\ndone in {step_idx} steps")
    print(f"final cache free: {cache.block_table.num_free}/{cache.num_blocks} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
