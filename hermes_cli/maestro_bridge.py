"""Bridge `hermes maestro ...` to the Hermes-Zen-Agent maestro package.

This module is the install-side companion to
`docs/upstream-patches/hermes-cli-maestro-bridge.md` in the
Hermes-Zen-Agent repo. It is loaded from `hermes_cli.main` only when
the maestro package is reachable on sys.path (or made reachable via
`MAESTRO_REPO_ROOT`); failures during registration are swallowed by
the caller so upstream installs without the maestro package keep
working unchanged.

Resolution order for the maestro package:
  1. Already importable (PYTHONPATH/CWD/site-packages).
  2. `MAESTRO_REPO_ROOT` env var → prepend to sys.path.
  3. `~/Hermes-Zen-Agent` (default location of the repo) →
     prepend to sys.path.

The shim never *runs* maestro code — it only wires up argparse so
`hermes maestro doctor --json` reaches `maestro.cli.maestro_command`.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _purge_stale_maestro(target_root: Path) -> None:
    """Drop any cached maestro.* modules loaded from a path other than
    `target_root`. Without this, an explicit MAESTRO_REPO_ROOT cannot
    win over a stale `import maestro.cli` that already succeeded.

    Uses path-containment (resolved) instead of substring matching so a
    sibling like `<target>-old/maestro/__init__.py` cannot slip past
    the purge. Also checks `maestro.cli` in addition to `maestro`."""
    target_resolved = target_root.resolve()
    for module_name in ("maestro", "maestro.cli"):
        loaded = sys.modules.get(module_name)
        if loaded is None:
            continue
        loaded_path_str = getattr(loaded, "__file__", None)
        if not loaded_path_str:
            continue
        try:
            loaded_resolved = Path(loaded_path_str).resolve()
            if loaded_resolved.is_relative_to(target_resolved):
                continue
        except (OSError, ValueError):
            pass
        for key in list(sys.modules.keys()):
            if key == "maestro" or key.startswith("maestro."):
                sys.modules.pop(key, None)
        return


def _cwd_walk_candidate() -> Path | None:
    """Walk up from cwd looking for a Hermes-Zen-Agent checkout. Requires
    both `maestro/cli.py` AND `.git/` so an unrelated `maestro/` package
    elsewhere on disk cannot hijack the bridge."""
    try:
        cwd = Path.cwd().resolve()
    except (FileNotFoundError, OSError):
        return None
    for root in [cwd, *cwd.parents]:
        if (root / "maestro" / "cli.py").is_file() and (root / ".git").exists():
            return root
    return None


def _ensure_maestro_importable() -> None:
    # Explicit env override wins — even over an already-imported stale
    # maestro.cli (codex R10 HIGH on PR #36: stale ~/Hermes-Zen-Agent
    # was silently winning over the operator's MAESTRO_REPO_ROOT).
    env_root = os.environ.get("MAESTRO_REPO_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
        if root.is_dir() and (root / "maestro" / "cli.py").is_file():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            _purge_stale_maestro(root)
        else:
            # Surface the operator error rather than silently fall back
            # to a stale checkout — the whole point of the env override
            # is to be authoritative.
            print(
                f"[hermes maestro bridge] MAESTRO_REPO_ROOT={env_root!r} "
                f"is not a Hermes-Zen-Agent checkout (missing maestro/cli.py). "
                f"Ignoring; resolution will fall through.",
                file=sys.stderr,
            )

    try:
        importlib.import_module("maestro.cli")
        return
    except ImportError:
        pass

    candidates: list[Path] = []
    cwd_root = _cwd_walk_candidate()
    if cwd_root is not None:
        candidates.append(cwd_root)
    candidates.append(Path.home() / "Hermes-Zen-Agent")

    for root in candidates:
        if not root.is_dir():
            continue
        if not (root / "maestro" / "cli.py").is_file():
            continue
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
        break

    importlib.import_module("maestro.cli")  # raises if still unreachable


def _dispatch(args) -> int | None:
    """Top-level argparse `func`. Hermes' main loop ignores the return
    value, so we sys.exit on non-zero to make `hermes maestro doctor`
    a real exit-code gate that deploy scripts can branch on."""
    from maestro.cli import maestro_command
    rc = maestro_command(args)
    if rc:
        sys.exit(rc)
    return rc


def register_maestro_subparser(subparsers) -> None:
    _ensure_maestro_importable()
    from maestro.cli import register
    register(subparsers, dispatch=_dispatch)
