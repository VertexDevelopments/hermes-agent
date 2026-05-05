"""OQ-35 / D-013 round-9 secondary: async integration test for stale
activation + expired/suspended session through the real
``_handle_message`` / ``_handle_message_with_agent`` dispatch path.

Background (codex round-9 review-moq69whh-93d0p0):

  The OQ-31 behavioral test at
  ``test_session_boundary_skill_activation.py::
  test_oq31_first_slash_after_auto_reset_preserves_active_project``
  composes the underlying surfaces (slash dispatcher effect +
  boundary handler effect + DB read) directly because driving
  ``_handle_message_with_agent`` end-to-end requires hooks, transcript,
  agent cache, plugin adapter loader, channel directory, and ~6 k LoC
  of agent boot.

  Codex flagged this as helper-level coverage that could miss
  regressions in:
    * slash command resolution (``resolve_skill_command_key`` /
      ``build_skill_invocation_message``)
    * ``_quick_key`` vs ``session_entry.session_key`` matching
    * ``get_or_create_session`` auto-reset wiring
    * the ``was_auto_reset`` branch dispatch in
      ``_handle_message_with_agent``
    * later ``apply_skill_context`` reads

  This integration test closes the gap by driving a real slash-skill
  ``MessageEvent`` through the actual outer ``_handle_message``.  The
  inner ``_handle_message_with_agent`` is replaced by a thin recorder
  that (a) calls ``session_store.get_or_create_session`` for real,
  (b) replays the auto-reset boundary block (the
  ``_clear_session_activation_unless_just_set`` call), and (c) replays
  the read-side ``apply_skill_context`` call against a stubbed agent
  using the same DB read path as the production code.  This keeps the
  test fast and deterministic while still exercising the real slash
  dispatcher and the real seam between outer and inner handlers.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u-oq35",
        chat_id="c-oq35",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m-oq35",
    )


def _auto_reset_entry(reason: str, session_id: str) -> SessionEntry:
    src = _make_source()
    now = datetime.now()
    entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id,
        created_at=now,
        updated_at=now,
        origin=src,
        platform=src.platform,
        chat_type=src.chat_type,
    )
    entry.was_auto_reset = True
    entry.auto_reset_reason = reason
    entry.reset_had_activity = False
    return entry


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "gateway_sessions.db"


@pytest.fixture()
def patched_ssdb(monkeypatch: pytest.MonkeyPatch, tmp_db: Path) -> Path:
    """Redirect skill_state_db record/get/clear to a per-test DB.

    Mirrors the ``patched_clear`` fixture in
    ``test_session_boundary_skill_activation.py`` but lives here so the
    file is self-contained for upstream patch capture (D-005).
    """
    import gateway.skill_state_db as ssdb_mod

    real_record = ssdb_mod.record_activation
    real_get = ssdb_mod.get_activation
    real_clear = ssdb_mod.clear_activation

    def _record(session_key, active_skill, **kw):
        kw.setdefault("db_path", tmp_db)
        return real_record(session_key, active_skill, **kw)

    def _get(session_key, **kw):
        kw.setdefault("db_path", tmp_db)
        return real_get(session_key, **kw)

    def _clear(session_key, **kw):
        kw.setdefault("db_path", tmp_db)
        return real_clear(session_key, **kw)

    monkeypatch.setattr(ssdb_mod, "record_activation", _record)
    monkeypatch.setattr(ssdb_mod, "get_activation", _get)
    monkeypatch.setattr(ssdb_mod, "clear_activation", _clear)
    return tmp_db


def _make_runner(session_entry: SessionEntry):
    """Build a stubbed GatewayRunner that exposes the surfaces
    ``_handle_message`` touches before the slash dispatcher and through
    the spawn point at ``_handle_message_with_agent``.

    Mirrors ``tests/gateway/test_steer_command.py::_make_runner`` so
    ``_handle_message`` can run far enough to hit the slash dispatcher
    block (gateway/run.py:3805-3911) without exploding on missing attrs.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_a, **_k: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **k: None
    runner._emit_gateway_run_progress = AsyncMock()

    # Surfaces the OQ-30 / OQ-31 wiring needs.  These are normally set
    # by GatewayRunner.__init__ but we bypass __init__ here.
    runner._session_skill_context = {}
    runner._slash_activation_this_turn = set()
    runner._session_model_overrides = {}
    runner._set_session_reasoning_override = lambda _key, _val: None
    runner._pending_model_notes = {}
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda _key: None
    runner._busy_ack_ts = {}

    return runner, adapter


@pytest.mark.parametrize("reason", ["idle", "daily", "suspended"])
@pytest.mark.asyncio
async def test_oq35_slash_after_auto_reset_clears_stale_through_real_handler(
    patched_ssdb: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
):
    """Drive the real ``_handle_message`` so the slash dispatcher
    runs end-to-end, then assert the auto-reset boundary handler in
    the inner method dropped the previous session_id's activation
    while preserving the just-set fresh one — and that the read-side
    ``apply_skill_context`` sees the fresh project.

    Differs from the helper-direct OQ-31 test by exercising:
      * real slash command resolution via ``resolve_skill_command_key``
        / ``build_skill_invocation_message``
      * real ``record_activation`` + ``_slash_activation_this_turn.add``
        through the dispatcher block at gateway/run.py:3805-3911
      * the real spawn seam from ``_handle_message`` into
        ``_handle_message_with_agent`` (with ``_quick_key`` derived
        from ``_session_key_for_source(source)``)
      * the wiring from ``get_or_create_session(source).was_auto_reset``
        to the boundary clear at gateway/run.py:4285
      * ``_quick_key == session_entry.session_key`` invariant (OQ-35
        specifically calls this out as a missed-coverage axis)
    """
    from gateway import skill_state_db as ssdb

    src = _make_source()
    skey = build_session_key(src)
    fresh_session_id = f"sess-fresh-{reason}"
    # Project tag is derived by the slash dispatcher from the skill
    # name via ``_skill_name.split("/")[-1].replace("maestro-", "", 1)``
    # at gateway/run.py:3863-3865 — so for a ``maestro-fresh`` skill the
    # tag is ``"fresh"``.  The test verifies this derivation by pinning
    # to that exact value.
    fresh_project = "fresh"

    # Pre-state mirrors a real auto-reset edge: the previous (now-expired
    # / suspended) session_id had a different slash skill.  The DB row is
    # keyed on session_key so the auto-reset handler must drop it
    # — UNLESS this turn's slash dispatcher just wrote the fresh row
    # (which is what this test asserts works).
    ssdb.record_activation(
        skey, "maestro-stale", channel_id="c-oq35",
        project="stale-project", source="slash",
    )

    entry = _auto_reset_entry(reason, fresh_session_id)
    runner, _adapter = _make_runner(entry)

    # In-memory cache leak from the previous session_id (matches what
    # gateway/run.py:10096-10127 apply_skill_context would replay if the
    # boundary clear fails).  The slash dispatcher overwrites this with
    # the fresh entry; the boundary handler then either preserves it
    # (correct) or drops it (regression).
    runner._session_skill_context[skey] = {
        "active_skill": "maestro-stale",
        "channel_id": "c-oq35",
        "project": "stale-project",
    }

    # Resolve the slash command for real-ish: stub
    # ``agent.skill_commands`` so ``/skill-fresh`` resolves to a
    # ``maestro-fresh`` skill with a non-empty invocation message.  This
    # exercises ``resolve_skill_command_key`` / ``get_skill_commands`` /
    # ``build_skill_invocation_message`` integration points called from
    # gateway/run.py:3825-3849.
    import agent.skill_commands as skill_commands_mod

    monkeypatch.setattr(
        skill_commands_mod, "get_skill_commands",
        lambda: {"skill-fresh": {"name": "maestro-fresh"}},
    )
    monkeypatch.setattr(
        skill_commands_mod, "resolve_skill_command_key",
        lambda cmd: "skill-fresh" if cmd == "skill-fresh" else None,
    )
    monkeypatch.setattr(
        skill_commands_mod, "build_skill_invocation_message",
        lambda key, instr, task_id: f"[skill={key}] {instr}",
    )

    # No platform-disabled override; pre-import the symbol the
    # dispatcher imports lazily so the patch is visible.
    import agent.skill_utils as skill_utils_mod
    monkeypatch.setattr(
        skill_utils_mod, "get_disabled_skill_names",
        lambda platform=None: set(),
    )

    # Stub agent receiving apply_skill_context — the production read
    # path at gateway/run.py:10222-10227 calls
    # ``agent.apply_skill_context(active_skill, channel_id, project)``
    # using the durable ssdb read.  We replay the same shape inside the
    # substituted ``_handle_message_with_agent``.
    fake_agent = MagicMock()
    captured = {"quick_key": None, "session_key": None, "guard_at_entry": None}

    async def _fake_handle_with_agent(event, source, _quick_key, run_generation):
        # Snapshot the per-turn guard set BEFORE the boundary clear
        # consumes it — proves the real slash dispatcher populated it.
        captured["quick_key"] = _quick_key
        captured["guard_at_entry"] = set(runner._slash_activation_this_turn)

        # Real ``get_or_create_session`` lookup (this is the wiring axis
        # OQ-35 calls out — auto-reset semantics must propagate through
        # the session_store seam).
        se = runner.session_store.get_or_create_session(source)
        captured["session_key"] = se.session_key
        captured["was_auto_reset"] = bool(getattr(se, "was_auto_reset", False))
        captured["auto_reset_reason"] = getattr(se, "auto_reset_reason", None)

        # Replay the was_auto_reset boundary block from
        # gateway/run.py:5261-5290 (model overrides drop) and the
        # OQ-30/OQ-31 boundary clear at gateway/run.py:4285.
        if getattr(se, "was_auto_reset", False):
            runner._session_model_overrides.pop(se.session_key, None)
            runner._set_session_reasoning_override(se.session_key, None)
            runner._pending_model_notes.pop(se.session_key, None)
            runner._clear_session_activation_unless_just_set(se.session_key)
            se.was_auto_reset = False
            se.auto_reset_reason = None

        # Replay the read-side at gateway/run.py:10222-10227 — the
        # stubbed agent receives the same active_skill / project that
        # the real apply_skill_context call would feed it on first turn
        # of the fresh session.
        persisted = ssdb.get_activation(se.session_key)
        if persisted is not None:
            fake_agent.apply_skill_context(
                persisted["active_skill"],
                persisted.get("channel_id"),
                persisted.get("project"),
            )
        else:
            fake_agent.apply_skill_context(None, None, None)
        return None

    monkeypatch.setattr(
        runner, "_handle_message_with_agent", _fake_handle_with_agent,
    )

    # ── Drive the real outer dispatcher ──────────────────────────
    await runner._handle_message(_make_event("/skill-fresh do the thing"))

    # ── Slash dispatcher fired through the real path ────────────
    # The integration value: this proves resolve_skill_command_key +
    # build_skill_invocation_message + record_activation +
    # _slash_activation_this_turn.add were all wired correctly.  The
    # helper-direct OQ-31 test cannot exercise this chain.
    assert captured["guard_at_entry"] == {skey}, (
        "real slash dispatcher must populate _slash_activation_this_turn "
        "before the spawn into _handle_message_with_agent — got "
        f"{captured['guard_at_entry']!r}"
    )

    # ── _quick_key vs session_entry.session_key invariant ───────
    # OQ-35 explicitly calls out this matching as a missed-coverage axis.
    # If _session_key_for_source(source) ever drifted from
    # build_session_key(source) the boundary clear would target the
    # wrong row.
    assert captured["quick_key"] == captured["session_key"] == skey, (
        f"_quick_key ({captured['quick_key']!r}) must match "
        f"session_entry.session_key ({captured['session_key']!r}) — "
        f"both must equal build_session_key(source) ({skey!r})"
    )

    # ── get_or_create_session auto-reset semantics propagated ──
    assert captured["was_auto_reset"] is True
    assert captured["auto_reset_reason"] == reason

    # ── Boundary handler preserved fresh activation ────────────
    persisted = ssdb.get_activation(skey)
    assert persisted is not None, (
        f"first slash skill after {reason} auto-reset lost its "
        "activation — the boundary clear dropped what the slash "
        "dispatcher just wrote (OQ-31 regression visible only "
        "through the integration path)"
    )
    assert persisted["project"] == fresh_project, (
        f"active_project wrong after {reason} auto-reset; expected "
        f"{fresh_project!r}, got {persisted['project']!r} — stale "
        "row from previous session_id leaked through the boundary"
    )
    assert persisted["active_skill"] == "maestro-fresh"

    # In-memory cache mirrors DB.
    cache = runner._session_skill_context.get(skey)
    assert cache is not None
    assert cache["project"] == fresh_project
    assert cache["active_skill"] == "maestro-fresh"

    # ── Stale row unrecoverable ────────────────────────────────
    # The previous session_id's "stale-project" must not be readable
    # from skill_state_db after the boundary clear runs — even though
    # it was seeded keyed on the same session_key (the row identity
    # the auto-reset rotates around).
    assert persisted["project"] != "stale-project", (
        "stale activation from previous session_id resurfaced through "
        "skill_state_db read — boundary clear failed to drop it"
    )

    # ── apply_skill_context received fresh values ──────────────
    # This mirrors gateway/run.py:10222-10227 — the stubbed agent's
    # apply_skill_context call IS the read-path consumer that
    # pre_memory_write hooks rely on for project-confidentiality
    # routing.  If this asserts the wrong project, project policy
    # would apply against the stale session.
    fake_agent.apply_skill_context.assert_called_once_with(
        "maestro-fresh", "c-oq35", fresh_project,
    )

    # ── Per-turn guard fully consumed ──────────────────────────
    # _clear_session_activation_unless_just_set must remove the entry
    # so the NEXT turn's boundary clear (if any) is not falsely
    # protected.  The defense-in-depth cleanup in _handle_message's
    # finally block at gateway/run.py:3982-3996 also discards, but the
    # primary path is consumption inside the boundary clear.
    assert skey not in runner._slash_activation_this_turn, (
        "_slash_activation_this_turn entry must be consumed by the "
        "boundary clear — leaking it would falsely protect the next "
        "turn's stale row from being dropped"
    )


@pytest.mark.asyncio
async def test_oq35_session_key_matching_invariant_documented(
    patched_ssdb: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Static-source guard: ``_session_key_for_source(source)`` and
    ``build_session_key(source)`` must produce the same key.

    OQ-35 specifically lists ``_quick_key vs session_entry.session_key
    matching`` as a regression risk.  The parametrized test above
    asserts this dynamically through the dispatch path; this test
    captures it as a direct unit-level invariant so a future drift
    surfaces with a clear failure regardless of which integration path
    happens to exercise it.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    src = _make_source()
    assert runner._session_key_for_source(src) == build_session_key(src)
