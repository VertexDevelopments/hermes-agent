"""Round-7 M (codex round-6 MEDIUM, conf 0.98): the migration script
``scripts/migrate_gateway_sessions.py`` is documented and shebang'd as a
direct-run CLI but imports ``hermes_constants`` at module top level.
Direct execution puts ``scripts/`` on ``sys.path`` (NOT the repo root),
so the import fails with ``ModuleNotFoundError: No module named
'hermes_constants'`` whenever the operator runs the script as documented:

    $ python3 scripts/migrate_gateway_sessions.py status --db /tmp/foo
    ModuleNotFoundError: No module named 'hermes_constants'

Fix: prepend the repo root to ``sys.path`` before the
``hermes_constants`` import so the script is self-contained for direct
invocation, with no PYTHONPATH gymnastics required.

This subprocess test invokes the script with an explicit ``cwd="/"`` and
a stripped ``env`` (no ``PYTHONPATH``) to prove the CLI works in the
exact configuration an operator hits in production.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "migrate_gateway_sessions.py"


def _system_python() -> str | None:
    """Locate a python interpreter that is NOT the test runner's venv.

    The repo is installed editable into the venv, so the venv python
    has ``hermes_constants`` resolvable via the .pth file even without
    the script's sys.path bootstrap — which silently masks the bug
    codex round-6 flagged.  The bug only manifests when an operator
    runs ``python3 scripts/migrate_gateway_sessions.py`` against a
    plain interpreter that has no editable-install path hook.

    Strategy: walk PATH for ``python3`` entries that don't live under
    the test venv prefix.  Skip the test if none is found (CI runners
    that only ship a venv'd python — this test is a regression guard,
    not load-bearing for correctness).
    """
    venv_prefix = sys.prefix  # The active venv when running tests.
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    seen: set[str] = set()
    for d in path_dirs:
        if not d:
            continue
        for name in ("python3", "python3.12", "python3.11", "python3.10"):
            cand = shutil.which(name, path=d)
            if not cand or cand in seen:
                continue
            seen.add(cand)
            try:
                resolved = os.path.realpath(cand)
            except OSError:
                continue
            # Skip anything inside the test venv.
            if resolved.startswith(os.path.realpath(venv_prefix)):
                continue
            # Confirm this python doesn't have hermes_constants on its
            # default sys.path (i.e., it's a "clean" interpreter that
            # would hit the bug without the script's sys.path fix).
            try:
                probe = subprocess.run(
                    [cand, "-c", "import hermes_constants"],
                    cwd="/",
                    env={"PATH": os.environ.get("PATH", "")},
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (subprocess.SubprocessError, OSError):
                continue
            if probe.returncode != 0 and "ModuleNotFoundError" in probe.stderr:
                return cand
    return None


def test_migrate_script_runs_standalone_without_pythonpath(tmp_path: Path):
    """Round-7 M: script must run via direct ``python3 path/to/script``
    invocation without PYTHONPATH set, from any cwd.  Without the
    sys.path bootstrap, this fails with ``ModuleNotFoundError: No
    module named 'hermes_constants'`` on a clean interpreter."""
    py = _system_python()
    if py is None:
        pytest.skip(
            "no clean python3 found on PATH outside the test venv; "
            "this regression guard requires an interpreter without "
            "the editable-install pth hook"
        )

    db_path = tmp_path / "round7-cli-smoke.db"

    # Strip PYTHONPATH so the script can't piggyback on the test
    # runner's environment.  Keep PATH (python may need to find
    # subprocesses) and HOME (for any get_hermes_home() default).
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
    }

    result = subprocess.run(
        [py, str(SCRIPT), "status", "--db", str(db_path)],
        # cwd="/" guarantees scripts/ is NOT silently on sys.path via
        # the test runner's working directory.
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "ModuleNotFoundError" not in result.stderr, (
        f"migrate_gateway_sessions.py raised ModuleNotFoundError when "
        f"run standalone (round-7 M leak):\nSTDERR:\n{result.stderr}"
    )
    assert result.returncode == 0, (
        f"migrate_gateway_sessions.py exited {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
