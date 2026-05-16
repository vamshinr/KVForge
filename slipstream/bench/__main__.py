"""Allow `python -m kvforge.bench` to invoke the benchmark CLI."""

from kvforge.bench.cli import main

raise SystemExit(main())
