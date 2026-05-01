"""Allow `python -m kvforge.optimize` to invoke the optimizer CLI."""

from kvforge.optimizer.cli import main

raise SystemExit(main())
