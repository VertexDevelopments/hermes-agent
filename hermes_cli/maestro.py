"""Bridge to the Maestro CLI module vendored in Hermes-Zen-Agent.

Per codex C1 finding, the CLI logic lives in Hermes-Zen-Agent so its
tests are hermetic against this fork install. The fork ships only this
bridge so `hermes maestro ...` works on the operator's machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MAESTRO_REPO = Path.home() / "Hermes-Zen-Agent"


def _ensure_repo_on_path() -> bool:
    if not (_MAESTRO_REPO / "maestro" / "cli.py").exists():
        return False
    repo_str = str(_MAESTRO_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return True


def _missing_repo_error() -> int:
    sys.stderr.write(
        f"error: Maestro repo not found at {_MAESTRO_REPO}\n"
        "Clone https://github.com/VertexDevelopments/Hermes-Zen-Agent "
        "there to enable Maestro.\n"
    )
    return 1


def register(subparsers, *, dispatch=None) -> None:
    """Wire the maestro subcommand group into Hermes' argparse tree."""
    if not _ensure_repo_on_path():
        # Register a no-op so `hermes maestro` doesn't crash; instead
        # prints the helpful install error.
        parser = subparsers.add_parser(
            "maestro", help="Continuous-development factory bridge (NOT INSTALLED)"
        )
        parser.set_defaults(func=lambda args: _missing_repo_error())
        return
    from maestro.cli import register as _register  # noqa: E402
    _register(subparsers, dispatch=dispatch)


def maestro_command(args) -> int:
    """Top-level dispatch shim. Hermes' main.cmd_maestro calls this."""
    if not _ensure_repo_on_path():
        return _missing_repo_error()
    from maestro.cli import maestro_command as _cmd  # noqa: E402
    return _cmd(args)
