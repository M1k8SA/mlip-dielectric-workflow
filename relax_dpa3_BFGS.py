#!/usr/bin/env python3
"""Backward-compatible entry point for the model-agnostic MLIP relaxer."""

from relax_mlip import main, relax_structure

__all__ = ["main", "relax_structure"]


if __name__ == "__main__":
    raise SystemExit(main())
