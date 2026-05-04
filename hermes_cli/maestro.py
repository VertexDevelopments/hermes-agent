"""`hermes maestro` subcommand group.

Wires the Maestro components in ~/Hermes-Zen-Agent/maestro/ + scripts/
to the Hermes CLI surface. Six verbs:

  hermes maestro plan [--projects ...] [--dry-run] [--post-discord]
  hermes maestro approve <program-id> [--all] [--force]
  hermes maestro status
  hermes maestro pause <program-id> [--reason TEXT]
  hermes maestro resume <program-id>
  hermes maestro abort <program-id> [--reason TEXT]

This module is thin — every verb dispatches to the maestro Python
package or scripts/maestro_planner.py. Logic lives there; this file
is parser plumbing + pretty-printing.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Maestro lives in the user's Hermes-Zen-Agent repo, not in the
# hermes-agent package. Add it to sys.path lazily so this module
# imports cleanly even when the repo is missing — `hermes maestro`
# will still print a useful error.
_MAESTRO_REPO = Path.home() / "Hermes-Zen-Agent"
_MAESTRO_DB = Path.home() / ".hermes" / "state" / "maestro.db"


def _ensure_repo_on_path() -> bool:
    """Returns True iff Hermes-Zen-Agent is importable. Caller prints
    a helpful error and returns nonzero if False."""
    if not (_MAESTRO_REPO / "maestro" / "__init__.py").exists():
        return False
    repo_str = str(_MAESTRO_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return True


def _missing_repo_error() -> int:
    sys.stderr.write(
        f"error: Maestro repo not found at {_MAESTRO_REPO}\n"
        "Clone https://github.com/VertexDevelopments/Hermes-Zen-Agent there to enable Maestro.\n"
    )
    return 1


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _cmd_plan(args) -> int:
    """Delegate to scripts/maestro_planner.py via subprocess so the
    planner runs under its own venv path resolution + sys.path setup."""
    if not _ensure_repo_on_path():
        return _missing_repo_error()
    cmd = [
        sys.executable,
        str(_MAESTRO_REPO / "scripts" / "maestro_planner.py"),
    ]
    if args.projects:
        cmd += ["--projects", args.projects]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.post_discord:
        cmd += ["--post-discord"]
    if args.db:
        cmd += ["--db", str(args.db)]
    return subprocess.run(cmd, check=False).returncode


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------

def _cmd_approve(args) -> int:
    if not _ensure_repo_on_path():
        return _missing_repo_error()
    from maestro import dispatcher  # noqa: E402

    db_path = args.db or _MAESTRO_DB

    if args.all:
        ids = _list_pending(db_path)
        if not ids:
            print("No pending approvals.")
            return 0
        rc = 0
        for pid in ids:
            try:
                r = dispatcher.approve(pid, db_path=db_path, force=args.force)
                print(f"approved: {r.program_id} ({r.project})")
            except dispatcher.DispatchError as e:
                sys.stderr.write(f"skipped {pid}: {e}\n")
                rc = 1
        return rc

    if not args.program_id:
        sys.stderr.write("error: program_id required (or pass --all)\n")
        return 2
    try:
        r = dispatcher.approve(args.program_id, db_path=db_path, force=args.force)
    except dispatcher.DispatchError as e:
        sys.stderr.write(f"error: {e}\n")
        return 1
    print(f"approved: {r.program_id} ({r.project})")
    print(f"program_dir: {r.program_dir}")
    print("driver loop will pick up at next 60s tick")
    return 0


def _list_pending(db_path: Path) -> list[str]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    try:
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = con.execute(
            "SELECT program_id FROM pending_approvals "
            "WHERE dispatched=0 AND expires_at > ? ORDER BY created_at",
            (now_iso,),
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def _cmd_status(args) -> int:
    if not _ensure_repo_on_path():
        return _missing_repo_error()
    from maestro import dispatcher  # noqa: E402

    db_path = args.db or _MAESTRO_DB
    if not db_path.exists():
        print("No maestro.db yet. Run `hermes maestro plan` to create one.")
        return 0
    print(dispatcher.status_summary(db_path))
    return 0


# ---------------------------------------------------------------------------
# pause / resume / abort
# ---------------------------------------------------------------------------

def _set_state(db_path: Path, program_id: str, *, target_state: str,
               reason: str | None, allowed_from: tuple[str, ...]) -> int:
    """Transition program state with operator-friendly errors."""
    if not db_path.exists():
        sys.stderr.write(f"error: maestro.db not found at {db_path}\n")
        return 1
    con = sqlite3.connect(str(db_path))
    try:
        with con:
            row = con.execute(
                "SELECT state FROM dispatched_programs WHERE program_id = ?",
                (program_id,),
            ).fetchone()
            if row is None:
                sys.stderr.write(f"error: no dispatched program {program_id}\n")
                return 1
            current = row[0]
            if current not in allowed_from:
                sys.stderr.write(
                    f"error: cannot transition {program_id} from {current} "
                    f"to {target_state} (allowed from: {', '.join(allowed_from)})\n"
                )
                return 1

            params: dict = {"new_state": target_state, "pid": program_id}
            sets = ["state = :new_state"]
            if target_state == "paused":
                sets.append("pause_reason = :reason")
                params["reason"] = reason or "operator"
            elif target_state == "running":
                sets.append("pause_reason = NULL")
            elif target_state == "aborted":
                sets.append(
                    "completed_at = :completed_at, "
                    "lease_owner = NULL, lease_expires_at = NULL"
                )
                params["completed_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                if reason:
                    sets.append("pause_reason = :reason")
                    params["reason"] = f"abort:{reason}"

            sql = (
                "UPDATE dispatched_programs SET "
                + ", ".join(sets)
                + " WHERE program_id = :pid"
            )
            con.execute(sql, params)
    finally:
        con.close()
    print(f"{program_id} → {target_state}")
    return 0


def _cmd_pause(args) -> int:
    db_path = args.db or _MAESTRO_DB
    return _set_state(
        db_path, args.program_id,
        target_state="paused", reason=args.reason,
        allowed_from=("running",),
    )


def _cmd_resume(args) -> int:
    db_path = args.db or _MAESTRO_DB
    return _set_state(
        db_path, args.program_id,
        target_state="running", reason=None,
        allowed_from=("paused",),
    )


def _cmd_abort(args) -> int:
    db_path = args.db or _MAESTRO_DB
    return _set_state(
        db_path, args.program_id,
        target_state="aborted", reason=args.reason,
        allowed_from=("running", "paused"),
    )


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def register(subparsers, *, dispatch=None) -> None:
    """Called from hermes_cli.main: register the `maestro` subcommand
    group on the top-level argparse subparsers object.

    `dispatch` is the func argparse calls when `hermes maestro ...`
    runs. Hermes' top-level dispatcher ignores return values, so the
    caller usually passes a thin wrapper that sys.exit()s on non-zero.
    Defaults to maestro_command for direct use in tests."""
    parser = subparsers.add_parser(
        "maestro",
        help="Continuous-development factory bridge",
        description=(
            "Manage Maestro daily plans and dispatched programs. "
            "Bridges Hermes/Discord to ZenFlow's Brain orchestrator."
        ),
    )
    sub = parser.add_subparsers(dest="maestro_command")
    parser.set_defaults(func=dispatch or maestro_command)

    p_plan = sub.add_parser("plan", help="Run daily planning ritual")
    p_plan.add_argument("--projects", default="",
                        help="Comma-separated subset; default = all configured")
    p_plan.add_argument("--dry-run", action="store_true")
    p_plan.add_argument("--post-discord", action="store_true")
    p_plan.add_argument("--db", type=Path)

    p_approve = sub.add_parser("approve", help="Approve a pending program")
    p_approve.add_argument("program_id", nargs="?")
    p_approve.add_argument("--all", action="store_true",
                           help="Approve every non-expired pending program")
    p_approve.add_argument("--force", action="store_true",
                           help="Bypass the worker accountant cap")
    p_approve.add_argument("--db", type=Path)

    p_status = sub.add_parser("status", help="Show pending + dispatched programs")
    p_status.add_argument("--db", type=Path)

    p_pause = sub.add_parser("pause", help="Pause a running program")
    p_pause.add_argument("program_id")
    p_pause.add_argument("--reason")
    p_pause.add_argument("--db", type=Path)

    p_resume = sub.add_parser("resume", help="Resume a paused program")
    p_resume.add_argument("program_id")
    p_resume.add_argument("--db", type=Path)

    p_abort = sub.add_parser("abort", help="Abort a program (terminal)")
    p_abort.add_argument("program_id")
    p_abort.add_argument("--reason")
    p_abort.add_argument("--db", type=Path)


def maestro_command(args) -> int:
    """Top-level `hermes maestro ...` dispatcher. argparse sets
    `args.maestro_command` to the verb."""
    verb = getattr(args, "maestro_command", None)
    if verb is None:
        sys.stderr.write(
            "usage: hermes maestro {plan|approve|status|pause|resume|abort}\n"
        )
        return 2
    handlers = {
        "plan": _cmd_plan,
        "approve": _cmd_approve,
        "status": _cmd_status,
        "pause": _cmd_pause,
        "resume": _cmd_resume,
        "abort": _cmd_abort,
    }
    handler = handlers.get(verb)
    if handler is None:
        sys.stderr.write(f"unknown maestro verb: {verb}\n")
        return 2
    return handler(args)
