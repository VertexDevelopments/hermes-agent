"""Durable backing store for gateway slash-skill session activations (OQ-28).

This module is the single source of truth for ~/.hermes/state/
gateway_sessions.db access from gateway runtime code.

Why: round-3 H2 attempted to recover slash-skill activation context by
regex-scanning the conversation transcript for a SLASH_SKILL_MARKER
substring after a gateway restart. Codex round-4 review (conf 0.87)
flagged that as a strict regression:

  - SPOOFABLE: a user message containing the marker text could forge
    an active_project, fooling the pre_memory_write gate into believing
    a confidential project context was active and steering writes
    accordingly.
  - LOSSY: gateway compression rewrites the transcript head and the
    real activation marker disappears; no recovery on long sessions.

The structural fix is to persist activations to durable, non-user-
controllable state and read from that state on first turn after
restart. Transcripts are no longer consulted for session/project
resolution.

Auto-heal: every public function calls _ensure_schema() so a missing
DB file (fresh install, manual delete) heals on first use rather
than crashing. Mirrors scripts/session_event.py:_ensure_schema().

Stdlib only (sqlite3); no Hermes core deps.
"""
from __future__ import annotations

import importlib.util
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_DB_PATH = Path.home() / ".hermes" / "state" / "gateway_sessions.db"

# Lazy-loaded migrate module (avoids circular imports; the migrate
# script is a top-level CLI tool, not a package member).
_MIGRATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "migrate_gateway_sessions.py"
)
_MIGRATE_SPEC = importlib.util.spec_from_file_location(
    "_gateway_skill_state_migrate", _MIGRATE_PATH,
)


def _ensure_schema(db_path: Path) -> None:
    """Idempotently create the schema if missing.

    Call at the top of every public function. CREATE TABLE IF NOT
    EXISTS / INSERT OR IGNORE makes this cheap. Guarantees the first
    read or write after a fresh install / file deletion self-heals
    instead of raising.
    """
    if _MIGRATE_SPEC is None or _MIGRATE_SPEC.loader is None:
        # Should never happen in a normal install; surface the bug.
        raise RuntimeError(
            f"cannot locate migrate_gateway_sessions.py at {_MIGRATE_PATH}"
        )
    mig = importlib.util.module_from_spec(_MIGRATE_SPEC)
    _MIGRATE_SPEC.loader.exec_module(mig)
    mig.init_db(db_path)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the standard PRAGMAs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=5.0)
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("PRAGMA foreign_keys = ON")
    return con


def record_activation(
    session_key: str,
    active_skill: str,
    *,
    channel_id: Optional[str] = None,
    project: Optional[str] = None,
    source: str = "slash",
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Upsert a slash-skill activation for `session_key`.

    `source` must be 'slash' or 'auto_skill' (CHECK constraint in
    schema). Caller is responsible for the value (slash command path
    vs. channel-bound auto_skill resolution).

    Idempotent: same (session_key, skill) pair re-recorded simply
    refreshes activated_at.
    """
    if source not in ("slash", "auto_skill"):
        raise ValueError(f"invalid source {source!r}; expected slash|auto_skill")
    if not session_key or not active_skill:
        # Defensive: don't write empty PKs / required fields.
        logger.debug(
            "[skill_state_db] skip empty record session_key=%r skill=%r",
            session_key, active_skill,
        )
        return
    _ensure_schema(db_path)
    con = _connect(db_path)
    try:
        with con:
            con.execute(
                """
                INSERT INTO slash_skill_activations
                  (session_key, active_skill, active_channel_id,
                   active_project, source, activated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_key) DO UPDATE SET
                  active_skill      = excluded.active_skill,
                  active_channel_id = excluded.active_channel_id,
                  active_project    = excluded.active_project,
                  source            = excluded.source,
                  activated_at      = CURRENT_TIMESTAMP
                """,
                (session_key, active_skill, channel_id, project, source),
            )
    finally:
        con.close()


def get_activation(
    session_key: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[dict]:
    """Return the persisted activation row for `session_key`, or None.

    Returned dict shape mirrors what _session_skill_context (in-memory
    cache) holds:
        {"active_skill": str,
         "channel_id":   Optional[str],
         "project":      Optional[str],
         "source":       str}
    """
    if not session_key:
        return None
    _ensure_schema(db_path)
    con = _connect(db_path)
    try:
        row = con.execute(
            """
            SELECT active_skill, active_channel_id, active_project, source
              FROM slash_skill_activations
             WHERE session_key = ?
            """,
            (session_key,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    return {
        "active_skill": row[0],
        "channel_id":   row[1],
        "project":      row[2],
        "source":       row[3],
    }


def clear_activation(
    session_key: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Delete the activation row for `session_key` (no-op if absent).

    Call from /clear /new /reset handlers — session boundary operations
    must drop the activation so the next turn starts fresh.
    """
    if not session_key:
        return
    _ensure_schema(db_path)
    con = _connect(db_path)
    try:
        with con:
            con.execute(
                "DELETE FROM slash_skill_activations WHERE session_key = ?",
                (session_key,),
            )
    finally:
        con.close()


def clear_all_for_channel(
    channel_id: str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Delete every activation row tied to `channel_id` (no-op if none).

    Provided for completeness / future channel-rebind flows; not wired
    into the gateway today (channel-level resets aren't a current
    surface). Kept here so the wrapper API is the single ownership
    point and runtime code never opens raw connections.
    """
    if not channel_id:
        return
    _ensure_schema(db_path)
    con = _connect(db_path)
    try:
        with con:
            con.execute(
                "DELETE FROM slash_skill_activations "
                "WHERE active_channel_id = ?",
                (channel_id,),
            )
    finally:
        con.close()
