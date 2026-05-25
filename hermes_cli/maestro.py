"""Bridge installed Hermes to Hermes-Zen-Agent's Maestro CLI.

The CLI implementation lives in /Users/zenflow/Hermes-Zen-Agent/maestro/cli.py.
This installed module only makes that checkout importable and re-exports the
real command symbols.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path("/Users/zenflow/Hermes-Zen-Agent")
_CLI_PATH = _REPO_ROOT / "maestro" / "cli.py"


def _load_cli():
    if not _CLI_PATH.is_file():
        raise ImportError(f"Maestro CLI not found at {_CLI_PATH}")

    root = str(_REPO_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root]
    sys.path.insert(0, root)

    expected = _CLI_PATH.resolve()
    loaded = sys.modules.get("maestro.cli")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != expected:
        sys.modules.pop("maestro.cli", None)
        sys.modules.pop("maestro", None)

    module = importlib.import_module("maestro.cli")
    source = Path(getattr(module, "__file__", "")).resolve()
    if source != expected:
        raise ImportError(f"Loaded Maestro CLI from {source}, expected {expected}")
    return module


_cli = _load_cli()
register = _cli.register
maestro_command = _cli.maestro_command

__all__ = ["register", "maestro_command"]
