"""A TinyLlama-shaped decoder-only transformer.

This is *not* a faithful TinyLlama-1.1B reproduction; it's a structurally
identical but smaller decoder with the same kernel mix (RMSNorm, RoPE, GQA
attention, SwiGLU MLP) so the profiler exposes the same hot spots without
requiring a HuggingFace download or a multi-GB checkpoint.

Key shape parameters match the TinyLlama-1.1B family:
  hidden_size = 2048, intermediate = 5632, n_heads = 32, n_kv_heads = 4,
  n_layers configurable (default 4 for fast profiling, 22 for full size).

For real LLM benchmarking, swap this for a HuggingFace model via
`kvforge.models.hf_loader` (see optional `[hf]` extras).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyLlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    n_heads: int = 32
    n_kv_heads: int = 4          # GQA: 8x query heads per KV head
    n_layers: int = 4            # small for fast profiling; bump for realism
    head_dim: int = 64           # = hidden_size / n_heads
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_seq_len: int = 4096


# ---------- Building blocks ----------


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm.

    Reference (eager) implementation. The optimization loop replaces the
    forward call with a Triton kernel; this class stays as the correctness
    oracle for tests.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, cast back to input dtype.
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32 * self.weight).to(in_dtype)


def precompute_rope_cache(
    seq_len: int, head_dim: int, theta: float = 10000.0, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)             # [seq_len, head_dim/2]
    cos = angles.cos().repeat_interleave(2, dim=-1)  # [seq_len, head_dim]
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a [batch, n_heads, seq, head_dim] tensor.

    Reference (eager) implementation — replaced by Triton kernel during opt.
    """
    # Pair adjacent dims and rotate: (x_even, x_odd) -> (x_even*cos - x_odd*sin, ...)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    """GQA with eager attention (KV cache aware).

    Matches Llama-2/3 / TinyLlama: separate q/k/v projections, KV heads
    repeated to query head count via expand-and-reshape.
    """

    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        H, Hk, D = self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        k = self.k_proj(x).view(B, T, Hk, D).transpose(1, 2)  # [B, Hk, T, D]
        v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_cache = (k, v)

        # Repeat KV heads to match query heads (GQA expansion).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention with causal mask. We use F.sdpa so
        # PyTorch picks Flash / mem-efficient backend automatically; the
        # baseline this kernel replaces in the optimization loop is the
        # explicit softmax(QK^T / sqrt(d)) variant.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out), new_cache


class SwiGLUMLP(nn.Module):
    """SwiGLU feedforward as in Llama / TinyLlama."""

    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.post_attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.input_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.post_attn_norm(x))
        return x, new_cache


class TinyLlama(nn.Module):
    """The full decoder."""

    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        if cfg.hidden_size != cfg.n_heads * cfg.head_dim:
            raise ValueError("hidden_size must equal n_heads * head_dim")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        cos, sin = precompute_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        # Buffers so they move with `.to(device)`.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        B, T = tokens.shape
        x = self.embed(tokens)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, cos, sin, cache)
            new_caches.append(new_cache)

        x = self.final_norm(x)
        return self.lm_head(x), new_caches


# ---------- Convenience builders ----------


def build_tinyllama(
    n_layers: int = 4,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "auto",
    tiny: bool = False,
) -> TinyLlama:
    """Construct a TinyLlama and move it to device.

    `tiny=True` produces a much smaller model (vocab 1k, hidden 256) suitable
    for CPU smoke tests; the regular config matches TinyLlama-1.1B's per-layer
    shape and requires a GPU for any reasonable performance.
    """
    if tiny:
        cfg = TinyLlamaConfig(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=704,
            n_heads=4,
            n_kv_heads=2,
            n_layers=n_layers,
            head_dim=64,
        )
    else:
        cfg = TinyLlamaConfig(n_layers=n_layers)
    model = TinyLlama(cfg)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device=device, dtype=dtype if device != "cpu" else torch.float32)
    return model


def make_forward_fn(
    batch_size: int = 1,
    seq_len: int = 512,
    mode: str = "prefill",
) -> Callable[[TinyLlama], torch.Tensor]:
    """Return a closure suitable for `ModelProfiler.profile`.

    `prefill` mode: feeds a fresh `seq_len` token sequence each call.
    `decode` mode: simulates autoregressive decode by feeding 1 token at a time
    on top of a precomputed KV cache of length `seq_len - 1`.
    """
    if mode not in ("prefill", "decode"):
        raise ValueError("mode must be 'prefill' or 'decode'")

    def _fn(model: TinyLlama) -> torch.Tensor:
        device = next(model.parameters()).device
        if mode == "prefill":
            tokens = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), device=device)
            logits, _ = model(tokens, kv_caches=None, start_pos=0)
            return logits
        # decode mode: build a one-shot KV cache then take a single decode step
        with torch.inference_mode():
            prefill_tokens = torch.randint(
                0, model.cfg.vocab_size, (batch_size, seq_len - 1), device=device
            )
            _, kv_caches = model(prefill_tokens, kv_caches=None, start_pos=0)
            next_tok = torch.randint(0, model.cfg.vocab_size, (batch_size, 1), device=device)
            logits, _ = model(next_tok, kv_caches=kv_caches, start_pos=seq_len - 1)
            return logits

    return _fn
