"""OQ-30 / D-013 round-5: clear durable slash-skill activation on logical
session boundaries.

Background:

  OQ-28 introduced ~/.hermes/state/gateway_sessions.db keyed on session_key.
  Round-5 codex flagged (HIGH, conf 0.88) that the gateway can switch the
  same session_key to a different session_id via /resume, /branch, and
  the compression_exhausted auto-reset path. Without an explicit clear
  at those boundaries, an active_project from the OLD session_id leaks
  into the NEW transcript on the next read.

  Fix (Option B): clear the activation row AND the in-memory
  _session_skill_context cache at every logical-session boundary.

  Boundary paths covered here:
    1. _handle_resume_command  (/resume <name>)
    2. _handle_branch_command  (/branch [name])
    3. compression_exhausted   (auto-reset inside the message-handling loop)

  /reset and /new were already covered by OQ-28 round-4 — a regression
  guard is included here for completeness.

  HYGIENE COMPRESSION (mid-turn _compress_context that rotates the
  transcript session_id) is NOT a boundary — same logical conversation
  same activation. Do NOT clear there.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u-oq30",
        chat_id="c-oq30",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m-oq30")


def _make_entry(session_id: str) -> SessionEntry:
    src = _make_source()
    return SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=src,
        platform=src.platform,
        chat_type=src.chat_type,
    )


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Per-test gateway_sessions.db so tests don't pollute ~/.hermes."""
    return tmp_path / "gateway_sessions.db"


@pytest.fixture()
def patched_clear(monkeypatch: pytest.MonkeyPatch, tmp_db: Path):
    """Force gateway code to write/read against tmp_db.

    The runner's helper does `from gateway import skill_state_db as _ssdb`
    at the call site, so monkeypatching the module's clear_activation
    function with a wrapper that injects db_path=tmp_db is the cleanest
    way to redirect.
    """
    import gateway.skill_state_db as ssdb_mod

    real_record = ssdb_mod.record_activation
    real_get = ssdb_mod.get_activation
    real_clear = ssdb_mod.clear_activation

    def _record(session_key, active_skill, **kw):
        kw["db_path"] = tmp_db
        return real_record(session_key, active_skill, **kw)

    def _get(session_key, **kw):
        kw["db_path"] = tmp_db
        return real_get(session_key, **kw)

    def _clear(session_key, **kw):
        kw["db_path"] = tmp_db
        return real_clear(session_key, **kw)

    monkeypatch.setattr(ssdb_mod, "record_activation", _record)
    monkeypatch.setattr(ssdb_mod, "get_activation", _get)
    monkeypatch.setattr(ssdb_mod, "clear_activation", _clear)
    return tmp_db


def _seed_activation(tmp_db: Path, session_key: str) -> None:
    """Write a slash-skill activation directly so we can verify the
    boundary handler clears it."""
    from gateway import skill_state_db as ssdb_mod
    # Bypass any monkeypatch by going through the underlying file API.
    ssdb_mod.record_activation.__wrapped__ if hasattr(
        ssdb_mod.record_activation, "__wrapped__"
    ) else None
    # Use the real (un-patched) record path explicitly with tmp_db.
    import importlib
    real_mod = importlib.import_module("gateway.skill_state_db")
    # The fixture patches in-place; call clear/record with explicit db_path.
    real_mod.record_activation(
        session_key, "maestro-stale", channel_id="c-oq30",
        project="stale-project", source="slash", db_path=tmp_db,
    )


# ---------------------------------------------------------------------------
# Helpers to build runners for each boundary handler.
# ---------------------------------------------------------------------------


def _make_resume_runner():
    from gateway.run import GatewayRunner

    src = _make_source()
    skey = build_session_key(src)
    current = _make_entry("current-session-A")
    resumed = _make_entry("resumed-session-B")

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._background_tasks = set()
    runner._async_flush_memories = AsyncMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._agent_cache_lock = None
    runner._session_skill_context = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current
    runner.session_store.switch_session.return_value = resumed
    runner.session_store.load_transcript.return_value = []
    runner._session_db = MagicMock()
    runner._session_db.resolve_session_by_title.return_value = "resumed-session-B"
    runner._session_db.get_session_title.return_value = "Resumed"
    return runner, skey


def _make_branch_runner():
    from gateway.run import GatewayRunner

    src = _make_source()
    skey = build_session_key(src)
    current = _make_entry("current-session-A")
    branched = _make_entry("branched-session-B")

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._agent_cache_lock = None
    runner._session_skill_context = {}
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    runner.session_store.switch_session.return_value = branched
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = "Current"
    runner._session_db.get_next_title_in_lineage.return_value = "Current #2"
    return runner, skey


# ---------------------------------------------------------------------------
# OQ-30 boundary clear tests.  Each MUST fail without the runtime fix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_clears_durable_slash_skill_activation(
    patched_clear: Path,
):
    """OQ-30: /resume MUST clear the slash-skill activation row + cache.
    Otherwise the resumed session_id reads back the previous session_id's
    active_project."""
    runner, skey = _make_resume_runner()
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-stale", "channel_id": None,
        "project": "stale-project",
    }
    _seed_activation(patched_clear, skey)

    from gateway import skill_state_db as ssdb
    assert ssdb.get_activation(skey) is not None

    await runner._handle_resume_command(_make_event("/resume Resumed"))

    assert skey not in runner._session_skill_context, (
        "in-memory _session_skill_context still holds stale activation "
        "after /resume — OQ-30 leak via cache"
    )
    assert ssdb.get_activation(skey) is None, (
        "DB row still present after /resume — OQ-30 leak via durable state"
    )


@pytest.mark.asyncio
async def test_branch_clears_durable_slash_skill_activation(
    patched_clear: Path,
):
    """OQ-30: /branch MUST clear the slash-skill activation row + cache."""
    runner, skey = _make_branch_runner()
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-stale", "channel_id": None,
        "project": "stale-project",
    }
    _seed_activation(patched_clear, skey)

    from gateway import skill_state_db as ssdb
    assert ssdb.get_activation(skey) is not None

    await runner._handle_branch_command(_make_event("/branch"))

    assert skey not in runner._session_skill_context, (
        "in-memory _session_skill_context still holds stale activation "
        "after /branch — OQ-30 leak via cache"
    )
    assert ssdb.get_activation(skey) is None, (
        "DB row still present after /branch — OQ-30 leak via durable state"
    )


def test_compression_exhausted_clears_durable_slash_skill_activation(
    patched_clear: Path,
):
    """OQ-30: the compression_exhausted auto-reset MUST clear the
    slash-skill activation.  We can't easily replay the full
    handle_message coroutine, so we exercise the runner-level helper
    directly — every boundary path must call it.

    The presence of the call inside the compression_exhausted branch is
    asserted by the source-level guard below; this test verifies the
    helper itself does the right thing.
    """
    from gateway.run import GatewayRunner
    from gateway import skill_state_db as ssdb

    src = _make_source()
    skey = build_session_key(src)

    runner = object.__new__(GatewayRunner)
    runner._session_skill_context = {
        skey: {"active_skill": "maestro-stale", "channel_id": None,
               "project": "stale-project"},
    }
    _seed_activation(patched_clear, skey)
    assert ssdb.get_activation(skey) is not None

    runner._clear_session_activation(skey)

    assert skey not in runner._session_skill_context
    assert ssdb.get_activation(skey) is None


def test_run_py_clears_activation_on_compression_exhausted_path():
    """Source-level regression guard: the compression_exhausted block in
    gateway/run.py must invoke _clear_session_activation (or
    skill_state_db.clear_activation) so the durable row is dropped.

    Without this, even with the helper present, the boundary site would
    silently reset the session without clearing the activation — the
    exact OQ-30 leak.
    """
    src = (
        Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    ).read_text(encoding="utf-8")

    # Locate the compression_exhausted branch (a few hundred chars).
    idx = src.find("compression_exhausted")
    assert idx != -1, "compression_exhausted block missing entirely"

    # Search the next ~1500 chars for evidence of the activation clear.
    window = src[idx: idx + 1500]
    assert (
        "_clear_session_activation" in window
        or "_ssdb.clear_activation" in window
    ), (
        "compression_exhausted block does NOT clear the slash-skill "
        "activation — OQ-30 leak across the auto-reset boundary"
    )
