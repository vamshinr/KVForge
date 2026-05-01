"""Allow `python -m kvforge.profile` to invoke the profiler CLI."""

from kvforge.profiler.cli import main

raise SystemExit(main())
