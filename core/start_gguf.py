"""GGUF entry point for the v6 runtime; --backbone chooses the base model."""

from __future__ import annotations

import sys

from .main import main as run_core


def main(argv: list[str] | None = None) -> None:
    options = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--backend" or arg.startswith("--backend=") for arg in options):
        raise SystemExit("GGUF starter fixes --backend gguf; use python -m core.main to choose a backend")
    run_core(["--backend", "gguf", *options])


if __name__ == "__main__":
    main()
