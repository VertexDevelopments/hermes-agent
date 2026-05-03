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


# ---------------------------------------------------------------------------
# Round-7 H1: idle / daily / suspended auto-reset path (was_auto_reset)
# ---------------------------------------------------------------------------
#
# Round-6 wired the boundary clear for /resume, /branch, and the
# compression_exhausted path, but missed the *other* auto-reset boundary:
# `get_or_create_session` rotating session_id under the same session_key
# when a session is idle, daily-rolled, or recovered as "suspended" after
# a crash.  The handler at gateway/run.py:4162-4222 (was_auto_reset block)
# detects this, posts the user notice, then clears the flags — but never
# drops the durable activation row.  Result: a stale active_project from
# the expired session_id is read back and applied to the fresh transcript
# the moment the next pre_memory_write hook fires.
#
# Two-pronged guard, mirroring the round-6 pattern:
#   1. Behavioral test exercising the helper after seeding cache+DB
#      activation — proves the helper does the right thing when called
#      from the auto-reset path (analogous to the compression_exhausted
#      helper test above).
#   2. Source-level guard verifying gateway/run.py actually invokes the
#      helper inside the was_auto_reset block (analogous to the
#      compression_exhausted source-level guard above).


def test_auto_reset_clears_durable_slash_skill_activation(
    patched_clear: Path,
):
    """Round-7 H1: idle/daily/suspended auto-reset MUST clear the
    slash-skill activation row + cache.  Without this, the next
    `pre_memory_write` after the auto-reset reads back the stale
    active_project from the expired session_id (the leak codex round-6
    flagged at 0.9 confidence)."""
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

    # The auto-reset boundary fires inside _handle_message_with_agent's
    # was_auto_reset branch; that branch's responsibility (post round-7)
    # is to invoke _clear_session_activation(session_key) before the next
    # turn's skill context is applied.  Exercise the helper directly to
    # prove the contract.
    runner._clear_session_activation(skey)

    assert skey not in runner._session_skill_context, (
        "in-memory _session_skill_context still holds stale activation "
        "after auto-reset — round-7 H1 leak via cache"
    )
    assert ssdb.get_activation(skey) is None, (
        "DB row still present after auto-reset — round-7 H1 leak via "
        "durable state"
    )


def test_run_py_clears_activation_on_was_auto_reset_path():
    """Round-7 H1 source-level regression guard: the was_auto_reset block
    in gateway/run.py must invoke _clear_session_activation (or
    skill_state_db.clear_activation) so the durable activation row is
    dropped before the fresh session's skill context is applied.

    Codex round-6 finding (HIGH, conf 0.9): the round-5 fix wired
    boundary clears for /resume, /branch, and compression_exhausted but
    missed this fourth auto-reset path.  Without the clear here, the
    persisted active_project from the expired session_id leaks into the
    auto-reset transcript via _session_skill_context / DB read-back.
    """
    src = (
        Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    ).read_text(encoding="utf-8")

    # Locate the was_auto_reset branch — the user-facing notice block
    # that ends with `session_entry.was_auto_reset = False`.
    idx = src.find("if getattr(session_entry, 'was_auto_reset', False):")
    assert idx != -1, (
        "was_auto_reset branch missing entirely — gateway/run.py "
        "structure changed; update this guard"
    )
    end = src.find("session_entry.was_auto_reset = False", idx)
    assert end != -1, (
        "was_auto_reset branch terminator missing; structure changed"
    )

    window = src[idx:end]
    assert (
        "_clear_session_activation" in window
        or "_ssdb.clear_activation" in window
        or "skill_state_db.clear_activation" in window
    ), (
        "was_auto_reset block does NOT clear the slash-skill activation "
        "— round-7 H1 leak across the idle/daily/suspended auto-reset "
        "boundary (codex round-6 HIGH conf 0.9)"
    )


# ---------------------------------------------------------------------------
# OQ-31 (D-013 round-7 regression): same-turn slash activation protection
# ---------------------------------------------------------------------------
#
# Round-7's unconditional clear at the was_auto_reset boundary fixed the
# OQ-30-style leak from the *expired* session_id, but introduced a new
# regression: when the FIRST message after idle/daily/suspended is a slash
# skill, the slash dispatcher in handle_message persists activation
# (cache + DB) BEFORE _handle_message_with_agent runs.  The auto-reset
# clear inside _handle_message_with_agent then deletes that fresh
# activation, and slash skills (unlike auto_skill / channel bindings)
# have no repopulator for the rest of the turn — the user's slash skill
# runs without active_project set.
#
# Codex round-7 finding (review-moq11qzm-jdrcx7, HIGH conf 0.92).
# Required test (per finding): first post-idle/daily/suspended message is
# a slash skill, assert active_project resolves correctly.
#
# Approach implemented (per AUDIT.md): per-turn guard set
# `_slash_activation_this_turn` populated by the slash dispatcher, consulted
# by `_clear_session_activation_unless_just_set` inside the
# was_auto_reset branch — the boundary clear is suppressed only for the
# session_key whose activation was set this very turn.


def _make_runner_with_guard():
    """Build a minimally-stubbed runner exposing the OQ-31 guard surface.

    We don't need the full gateway plumbing — just the methods under
    test (`_clear_session_activation_unless_just_set` and
    `_clear_session_activation`) plus the in-memory cache and the per-turn
    set.  The DB call inside `_clear_session_activation` is monkeypatched
    in tests via the `patched_clear` fixture.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_skill_context = {}
    runner._slash_activation_this_turn = set()
    return runner


def test_oq31_unless_just_set_preserves_same_turn_activation(
    patched_clear: Path,
):
    """OQ-31 core: the same-turn-protection guard preserves the
    activation row + cache when session_key is in
    `_slash_activation_this_turn`, and consumes the entry."""
    from gateway import skill_state_db as ssdb

    src = _make_source()
    skey = build_session_key(src)

    runner = _make_runner_with_guard()
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-just-set",
        "channel_id": None,
        "project": "just-set-project",
    }
    _seed_activation(patched_clear, skey)
    # Re-seed using the just-set values so the assertion below checks
    # what the slash dispatcher would have written.
    ssdb.record_activation(
        skey, "maestro-just-set",
        channel_id=None, project="just-set-project",
        source="slash",
    )
    runner._slash_activation_this_turn.add(skey)

    cleared = runner._clear_session_activation_unless_just_set(skey)

    assert cleared is False, "guard MUST report 'preserved' for same-turn writer"
    assert skey not in runner._slash_activation_this_turn, (
        "guard MUST consume the per-turn set entry to avoid leaking into next turn"
    )
    assert skey in runner._session_skill_context, (
        "in-memory activation MUST survive auto-reset clear when set this turn"
    )
    assert runner._session_skill_context[skey]["project"] == "just-set-project"

    persisted = ssdb.get_activation(skey)
    assert persisted is not None, (
        "DB row MUST survive auto-reset clear when activation was set this turn"
    )
    assert persisted["project"] == "just-set-project"
    assert persisted["active_skill"] == "maestro-just-set"


def test_oq31_unless_just_set_clears_when_not_in_set(
    patched_clear: Path,
):
    """OQ-31 negative: when no slash activation was written this turn,
    the guard MUST fall through to the unconditional clear (preserving
    OQ-30 / round-6 behavior — drop the stale row from the previous
    session_id)."""
    from gateway import skill_state_db as ssdb

    src = _make_source()
    skey = build_session_key(src)

    runner = _make_runner_with_guard()
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-stale",
        "channel_id": None,
        "project": "stale-project",
    }
    _seed_activation(patched_clear, skey)
    assert ssdb.get_activation(skey) is not None
    # Note: _slash_activation_this_turn intentionally empty.

    cleared = runner._clear_session_activation_unless_just_set(skey)

    assert cleared is True, "guard MUST clear when no same-turn writer marked"
    assert skey not in runner._session_skill_context
    assert ssdb.get_activation(skey) is None


def test_oq31_unless_just_set_handles_empty_session_key():
    """Defensive: empty session_key returns False (preserved=False but
    nothing happened)."""
    runner = _make_runner_with_guard()
    assert runner._clear_session_activation_unless_just_set("") is False


def test_oq31_run_py_uses_guard_in_was_auto_reset_block():
    """Source-level regression guard for OQ-31: the was_auto_reset block
    must call the SAME-TURN-AWARE helper, not the unconditional clear.

    This is the structural ratchet that catches a future round-N
    regression where someone reverts to `_clear_session_activation`
    inside the was_auto_reset block.  Distinct from
    `test_run_py_clears_activation_on_was_auto_reset_path` which only
    checks substring "_clear_session_activation" (matches both helper
    names).  Here we assert the guarded form explicitly.
    """
    src = (
        Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    ).read_text(encoding="utf-8")

    idx = src.find("if getattr(session_entry, 'was_auto_reset', False):")
    end = src.find("session_entry.was_auto_reset = False", idx)
    assert idx != -1 and end != -1, (
        "was_auto_reset block bookends missing; structure changed"
    )
    window = src[idx:end]
    assert "_clear_session_activation_unless_just_set" in window, (
        "was_auto_reset block uses unconditional `_clear_session_activation` "
        "— OQ-31 regression: slash-skill activation set THIS turn would be "
        "deleted before pre_memory_write reads it.  Use "
        "`_clear_session_activation_unless_just_set` instead."
    )


def test_oq31_slash_dispatcher_marks_same_turn_set():
    """Source-level: the slash-skill dispatcher block must populate
    `_slash_activation_this_turn` after writing activation, otherwise
    the guard above is a no-op for real slash messages."""
    src = (
        Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    ).read_text(encoding="utf-8")

    # Find the slash-skill activation cache write.
    idx = src.find('_skill_ctx_dict[_quick_key] = {')
    assert idx != -1, "slash-skill cache write missing; structure changed"
    # Search forward to record_activation call.
    rec_idx = src.find("_ssdb.record_activation(", idx)
    assert rec_idx != -1, "slash-skill DB record call missing"
    window = src[idx:rec_idx + 800]
    assert "_slash_activation_this_turn.add(_quick_key)" in window, (
        "slash-skill dispatcher does NOT populate _slash_activation_this_turn "
        "after the cache/DB write — OQ-31 guard is unreachable for real "
        "slash invocations"
    )


def test_oq31_init_creates_per_turn_set():
    """Smoke: GatewayRunner.__init__ must declare
    `_slash_activation_this_turn` as a set so the guard can be consulted
    even on the very first message after process startup."""
    src = (
        Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    ).read_text(encoding="utf-8")
    assert "self._slash_activation_this_turn: set[str] = set()" in src, (
        "GatewayRunner.__init__ does not initialize "
        "_slash_activation_this_turn; the guard cannot fire"
    )


@pytest.mark.parametrize("reason", ["idle", "daily", "suspended"])
def test_oq31_first_slash_after_auto_reset_preserves_active_project(
    patched_clear: Path, reason: str,
):
    """OQ-31 behavioral test (codex round-7 required): when the FIRST
    message after idle / daily / suspended auto-reset is a slash skill,
    `active_project` resolves correctly for `pre_memory_write` because
    the same-turn-protection guard preserves the just-set activation.

    We compose the behavior from the underlying surfaces rather than
    invoking `_handle_message_with_agent` end-to-end (which requires
    hooks, config, transcript, agent cache, etc.):
      1. Slash dispatcher effect: write activation to cache+DB and add
         session_key to `_slash_activation_this_turn`.
      2. Auto-reset boundary effect: invoke
         `_clear_session_activation_unless_just_set(session_key)`.
      3. Read path: assert `active_project` is the new project, not None
         and not the previous session's stale project.

    Parametrized over `auto_reset_reason` to cover all three triggers
    (idle, daily, suspended) since the boundary handler treats them
    identically wrt the activation clear.
    """
    from gateway import skill_state_db as ssdb

    src = _make_source()
    skey = build_session_key(src)

    # Pre-state: previous (now-expired) session had a different slash skill.
    # The auto-reset rotated session_id but the row in the DB is keyed on
    # session_key, so it would normally be the stale row to clear.
    runner = _make_runner_with_guard()
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-stale",
        "channel_id": None,
        "project": "stale-project",
    }
    ssdb.record_activation(
        skey, "maestro-stale",
        channel_id=None, project="stale-project", source="slash",
    )

    # Simulate the slash dispatcher of THIS turn (handle_message branch
    # at gateway/run.py:3805-3911) — runs BEFORE the auto-reset clear.
    # The user invoked /skill-fresh, so cache + DB get the new row.
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-fresh",
        "channel_id": None,
        "project": f"fresh-project-{reason}",
    }
    ssdb.record_activation(
        skey, "maestro-fresh",
        channel_id=None, project=f"fresh-project-{reason}", source="slash",
    )
    runner._slash_activation_this_turn.add(skey)

    # Now _handle_message_with_agent reaches the was_auto_reset block
    # and calls the guard — it MUST preserve the writer-1 activation.
    runner._clear_session_activation_unless_just_set(skey)

    # Read path mirrors gateway/run.py:10096-10127 (apply_skill_context).
    persisted = ssdb.get_activation(skey)
    assert persisted is not None, (
        f"first slash skill after {reason} auto-reset lost its activation row "
        "— OQ-31 regression"
    )
    assert persisted["project"] == f"fresh-project-{reason}", (
        f"active_project wrong after {reason} auto-reset; expected fresh, got "
        f"{persisted['project']!r}"
    )
    assert persisted["active_skill"] == "maestro-fresh"
    cache = runner._session_skill_context.get(skey)
    assert cache is not None
    assert cache["project"] == f"fresh-project-{reason}"


def test_oq31_handle_message_finally_clears_per_turn_set():
    """Source-level: handle_message's outer `finally:` MUST discard the
    `_slash_activation_this_turn` entry as defense-in-depth, so a crash
    inside `_handle_message_with_agent` (before the auto-reset branch
    consumes the entry) cannot leak the marker into the next turn and
    falsely protect a genuine boundary clear."""
    src = (
        Path(__file__).resolve().parents[2] / "gateway" / "run.py"
    ).read_text(encoding="utf-8")

    # Locate the handle_message finally block: anchored on the unique
    # comment immediately preceding the agent-claim sentinel cleanup.
    anchor = "# If _run_agent replaced the sentinel with a real agent and"
    idx = src.find(anchor)
    assert idx != -1, "handle_message finally anchor missing; structure changed"
    # Search the next ~1500 chars (one finally block) for the discard.
    window = src[idx: idx + 1500]
    assert "_slash_activation_this_turn" in window, (
        "handle_message finally: does not discard _slash_activation_this_turn "
        "— a crash inside _handle_message_with_agent could leak the "
        "marker into the next turn"
    )
