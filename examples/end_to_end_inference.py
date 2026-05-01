"""Example: run a full prefill + decode loop on TinyLlama.

This is a minimal autoregressive generation showing how KVForge's components
integrate with a real inference pipeline.

Usage:
    python examples/end_to_end_inference.py
"""

from __future__ import annotations

import time

import torch

from kvforge.models.tinyllama import build_tinyllama


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tinyllama(n_layers=4, tiny=(device.type == "cpu"))
    model.eval()

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Prefill
    prompt_len = 128 if device.type == "cuda" else 32
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)

    print(f"\nPrefilling {prompt_len} tokens...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits, kv_caches = model(prompt, kv_caches=None, start_pos=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"  prefill: {prefill_ms:.1f} ms ({prompt_len / (prefill_ms / 1000):.0f} tok/s)")

    # Phase 2: Decode (greedy, just for timing)
    n_decode = 64 if device.type == "cuda" else 16
    print(f"\nDecoding {n_decode} tokens...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        cur_pos = prompt_len
        next_tok = logits[:, -1:].argmax(dim=-1)
        for _ in range(n_decode):
            logits, kv_caches = model(next_tok, kv_caches=kv_caches, start_pos=cur_pos)
            next_tok = logits[:, -1:].argmax(dim=-1)
            cur_pos += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) * 1000
    print(f"  decode: {decode_ms:.1f} ms total, "
          f"{decode_ms / n_decode:.2f} ms/tok ({1000 * n_decode / decode_ms:.1f} tok/s)")


if __name__ == "__main__":
    main()
