#!/usr/bin/env python3.12
"""Idempotent SQLite initializer for the gateway slash-skill activations DB.

OQ-28 (D-013 boundary hardening, codex round-4 recommendation): persist
gateway slash-skill activations to durable, non-user-controllable session
state so that

  (a) restart recovery does not depend on transcript-regex inference
      (which round-3 H2 attempted and round-4 codex rejected as spoofable
      and lossy under compression), and
  (b) the active_skill / active_project values fed into pre_memory_write
      hooks come from a trusted structured store, not from user-supplied
      message text.

Schema is intentionally separate from PM Shape C v1 (~/.hermes/state/
sessions.db). PM tracks Claude Code lifecycle hooks; this DB tracks
gateway runtime session->skill activations. Different concept, different
file, different lifecycle.

CLI:
  init    Create / migrate ~/.hermes/state/gateway_sessions.db (or --db PATH)
  status  Print schema_meta version, row counts, WAL/SHM presence

Module API:
  init_db(db_path: Path) -> None
      Idempotent. Runs the full DDL inside one transaction. Safe to call
      repeatedly; uses CREATE TABLE IF NOT EXISTS / INSERT OR IGNORE.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home


def _default_db_path() -> Path:
    """Resolve the gateway_sessions.db path under the active HERMES_HOME.

    Resolved at call time so multi-profile users (HERMES_HOME pointing
    at a per-profile dir) initialize / inspect the per-profile DB.
    OQ-29 (codex round-5 HIGH).
    """
    return get_hermes_home() / "state" / "gateway_sessions.db"


# Backwards-compatible name for callers that import the constant.
# DO NOT use as a function default — see _default_db_path().
DEFAULT_DB_PATH = _default_db_path()


# Full DDL. Kept as one string so init_db can run it in a single
# transaction. PRAGMAs run separately (they don't participate in the txn).
_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '1');

CREATE TABLE IF NOT EXISTS slash_skill_activations (
  session_key       TEXT PRIMARY KEY,
  active_skill      TEXT NOT NULL,
  active_channel_id TEXT,
  active_project    TEXT,
  activated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  source            TEXT NOT NULL CHECK (source IN ('slash','auto_skill'))
);

CREATE INDEX IF NOT EXISTS idx_activated_ts
  ON slash_skill_activations(activated_at DESC);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the standard PRAGMAs.

    WAL mode is per-database-file but we set it on every connection
    defensively (cheap, survives a fresh DB file appearing).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=5.0)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(db_path: Optional[Path] = None) -> None:
    """Create the DB + schema if missing. Idempotent.

    When `db_path` is omitted, resolves to ``HERMES_HOME/state/
    gateway_sessions.db`` at call time (OQ-29 multi-profile safety).
    """
    if db_path is None:
        db_path = _default_db_path()
    con = _connect(db_path)
    try:
        with con:
            con.executescript(_DDL)
    finally:
        con.close()


def status(db_path: Optional[Path] = None) -> dict:
    """Return a small dict useful for sanity checks.

    When `db_path` is omitted, resolves to ``HERMES_HOME/state/
    gateway_sessions.db`` at call time (OQ-29).
    """
    if db_path is None:
        db_path = _default_db_path()
    out: dict = {"db_path": str(db_path), "exists": db_path.exists()}
    if not db_path.exists():
        return out
    con = sqlite3.connect(str(db_path))
    try:
        ver = con.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        out["schema_version"] = ver[0] if ver else None
        out["activations_count"] = con.execute(
            "SELECT COUNT(*) FROM slash_skill_activations"
        ).fetchone()[0]
    finally:
        con.close()
    out["wal_present"] = (db_path.with_name(db_path.name + "-wal")).exists()
    out["shm_present"] = (db_path.with_name(db_path.name + "-shm")).exists()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Initialize / inspect the gateway slash-skill activations DB.",
    )
    p.add_argument("command", choices=("init", "status"))
    # Resolve default at parse time (NOT at module import) so HERMES_HOME
    # changes between import and invocation are honoured. OQ-29.
    _default_for_help = _default_db_path()
    p.add_argument("--db", type=Path, default=None,
                   help=f"DB path (default: {_default_for_help})")
    args = p.parse_args(argv)
    if args.db is None:
        args.db = _default_db_path()

    if args.command == "init":
        init_db(args.db)
        sys.stderr.write(f"initialized {args.db}\n")
        return 0
    if args.command == "status":
        s = status(args.db)
        for k, v in s.items():
            sys.stdout.write(f"{k}: {v}\n")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
