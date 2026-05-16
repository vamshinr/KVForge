"""``slipstream-serve`` — a tiny CLI that runs the engine on a prompt file.

This is **not** a serving stack — no HTTP, no streaming, no tokenizer server.
It's a smoke harness for the engine: feed N prompts in, get N completions out,
exit. Useful for debugging the end-to-end path and for the bench adapter's
``Engine.from_model_id`` once that lands.
"""

from __future__ import annotations

import argparse
import sys

from slipstream.engine.engine import Engine


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="slipstream-serve")
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--prompts-file", required=True,
                   help="Plain-text prompts, one per line.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--fp8-kv", action="store_true")
    args = p.parse_args(argv)

    try:
        engine = Engine.from_model_id(args.model, dtype=args.dtype, fp8_kv=args.fp8_kv)
    except NotImplementedError as e:
        print(f"engine not ready yet: {e}", file=sys.stderr)
        return 2

    with open(args.prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Submit + drain (left as exercise once tokenizer/wiring lands).
    print(f"would run {len(prompts)} prompts through {args.model} on engine={engine}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
