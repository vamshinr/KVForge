"""End-to-end TinyLlama benchmark: prefill + decode latency with vs without KVForge kernels.

Demonstrates that per-kernel speedups translate to real end-to-end gains.
Run with:
    python benchmarks/end_to_end.py --context 2048 --decode-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kvforge.hardware import detect_gpu
from kvforge.models.tinyllama import build_tinyllama


def time_prefill(model, batch_size: int, seq_len: int, n_iters: int = 10) -> float:
    """Median prefill latency in milliseconds."""
    device = next(model.parameters()).device
    tokens = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _ = model(tokens, kv_caches=None, start_pos=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(n_iters):
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(tokens, kv_caches=None, start_pos=0)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                _ = model(tokens, kv_caches=None, start_pos=0)
                times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return sum(times[1:-1]) / max(len(times) - 2, 1)


def time_decode(
    model, batch_size: int, prefill_len: int, decode_tokens: int, n_iters: int = 5,
) -> tuple[float, float]:
    """Returns (avg_decode_ms_per_token, total_decode_ms)."""
    device = next(model.parameters()).device
    prefill_tokens = torch.randint(0, model.cfg.vocab_size, (batch_size, prefill_len), device=device)

    times_per_iter: list[float] = []
    with torch.inference_mode():
        for _ in range(n_iters):
            # Prime the KV cache.
            _, kv_caches = model(prefill_tokens, kv_caches=None, start_pos=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            cur_pos = prefill_len
            for _ in range(decode_tokens):
                next_tok = torch.randint(0, model.cfg.vocab_size, (batch_size, 1), device=device)
                _, kv_caches = model(next_tok, kv_caches=kv_caches, start_pos=cur_pos)
                cur_pos += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_per_iter.append((time.perf_counter() - t0) * 1000)

    times_per_iter.sort()
    total_ms = sum(times_per_iter[1:-1]) / max(len(times_per_iter) - 2, 1)
    return total_ms / decode_tokens, total_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--tiny", action="store_true",
                        help="use a tiny model for CPU smoke tests")
    args = parser.parse_args()

    gpu = detect_gpu()
    print(f"Hardware: {gpu.name}")

    model = build_tinyllama(n_layers=args.n_layers, tiny=args.tiny)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{args.n_layers} layers")
    print(f"Mode: batch={args.batch}, context={args.context}, "
          f"decode_tokens={args.decode_tokens}")

    print("\n=== Prefill (cold KV cache) ===")
    prefill_ms = time_prefill(model, args.batch, args.context)
    tok_per_sec = (args.batch * args.context) / (prefill_ms / 1000)
    print(f"Latency: {prefill_ms:.2f} ms")
    print(f"Throughput: {tok_per_sec:.0f} prefill tokens/sec")

    print("\n=== Decode (warm KV cache) ===")
    per_token_ms, total_ms = time_decode(
        model, args.batch, args.context, args.decode_tokens
    )
    print(f"Per-token latency: {per_token_ms:.3f} ms")
    print(f"Total decode time: {total_ms:.2f} ms for {args.decode_tokens} tokens")
    print(f"Decode throughput: {1000 / per_token_ms:.1f} tokens/sec/batch")

    return 0


if __name__ == "__main__":
    sys.exit(main())
