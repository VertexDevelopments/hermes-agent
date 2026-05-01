"""Tests for the sticky typing loop + ``record_voice`` chat-action wiring.

Background — OPEN-QUESTIONS.md #23:
    Telegram's ``sendChatAction`` decays server-side after ~5s, so a single
    one-shot call at the start of a long-running response (LLM stream, voice
    transcription, Dagster pipeline trigger) leaves the user staring at a
    silent chat. ``BasePlatformAdapter._keep_typing`` already refreshes the
    indicator every ~2s; this PR extends it with a per-chat *action* override
    stack so a long-running step can flip the indicator from ``"typing"`` to
    ``"record_voice"`` (or any other Telegram chat-action) and have the
    refresh loop pick up the change immediately, without restarting the task.

The tests below stay fully offline — every Telegram HTTP call is mocked.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Adapter factory — bypasses ``__init__`` so we don't need real config or a
# live bot. Mirrors the pattern in ``test_telegram_thread_fallback.py``.
# ---------------------------------------------------------------------------


def _make_telegram_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._reply_to_mode = "first"
    # Fields the base ``__init__`` would normally set up; ``_keep_typing``
    # and ``_typing_action_scope`` need them.
    adapter._typing_paused = set()
    adapter._typing_action_overrides = {}
    adapter._active_sessions = {}
    return adapter


def _attach_chat_action_recorder(adapter) -> List[Dict[str, Any]]:
    """Replace ``adapter._bot.send_chat_action`` with a recorder that captures
    every call's kwargs. Returns the call log so the test can assert on it.
    """
    call_log: List[Dict[str, Any]] = []

    async def _recorder(**kwargs):
        call_log.append(dict(kwargs))

    adapter._bot = SimpleNamespace(send_chat_action=_recorder)
    return call_log


# ---------------------------------------------------------------------------
# 1. Sticky typing loop fires repeatedly while a long task runs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sticky_typing_loop_fires_repeatedly():
    """``_keep_typing`` must keep firing ``send_typing`` on its interval —
    not just once. Asserts at least two ticks (start + one refresh) within
    a simulated long-running response, proving the loop survives past the
    Telegram server-side ~5s decay window.
    """
    adapter = _make_telegram_adapter()
    call_log = _attach_chat_action_recorder(adapter)

    # interval=0 keeps the wall-clock cost of the test trivial; the relevant
    # behaviour (loop body re-fires until cancelled) is independent of the
    # exact sleep duration.
    task = asyncio.create_task(
        adapter._keep_typing("123", interval=0)
    )
    # Yield enough times for several iterations of the loop body to run.
    for _ in range(5):
        await asyncio.sleep(0)

    task.cancel()
    # ``_keep_typing`` swallows CancelledError internally (load-bearing for
    # normal handler completion), so awaiting the task must NOT re-raise.
    await task

    assert len(call_log) >= 2, (
        f"expected sticky loop to fire at least twice, got {len(call_log)}"
    )
    # Every tick should have used the default chat-action.
    for entry in call_log:
        assert entry["action"] == "typing"
        assert entry["chat_id"] == 123


# ---------------------------------------------------------------------------
# 2. Cancel cleanly — no stray sends after cancellation, no unhandled error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sticky_typing_cancels_cleanly():
    """Cancelling the task must stop further ``send_chat_action`` calls and
    must not raise — the existing ``except CancelledError: pass`` clause is
    load-bearing for normal handler completion.
    """
    adapter = _make_telegram_adapter()
    call_log = _attach_chat_action_recorder(adapter)

    task = asyncio.create_task(
        adapter._keep_typing("123", interval=0)
    )
    # Let the loop tick once, then cancel.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    calls_before_cancel = len(call_log)
    assert calls_before_cancel >= 1

    task.cancel()
    # ``_keep_typing`` swallows CancelledError; awaiting the task must
    # therefore complete normally rather than re-raising.
    await task

    # Give the loop a chance to do anything wrong post-cancel — it shouldn't.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(call_log) == calls_before_cancel, (
        "send_chat_action fired after the loop was cancelled"
    )


# ---------------------------------------------------------------------------
# 3. ``record_voice`` during transcription, then back to ``typing``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_voice_then_typing():
    """The action override stack lets a transcribe step flip the indicator
    to ``"record_voice"`` mid-flight, and exiting the scope reverts to
    ``"typing"`` — all without restarting the refresh task.
    """
    adapter = _make_telegram_adapter()
    call_log = _attach_chat_action_recorder(adapter)

    task = asyncio.create_task(
        adapter._keep_typing("123", interval=0)
    )
    # One tick of baseline "typing".
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    async with adapter._typing_action_scope("123", "record_voice"):
        # Several ticks while "transcribing".
        for _ in range(4):
            await asyncio.sleep(0)

    # Several ticks back on the default action.
    for _ in range(4):
        await asyncio.sleep(0)

    task.cancel()
    # ``_keep_typing`` swallows CancelledError internally (load-bearing for
    # normal handler completion), so awaiting the task must NOT re-raise.
    await task

    actions = [e["action"] for e in call_log]
    assert "record_voice" in actions, (
        f"expected at least one record_voice tick, got actions={actions!r}"
    )
    assert "typing" in actions, (
        f"expected at least one typing tick, got actions={actions!r}"
    )

    # The first record_voice tick must come after at least one typing tick,
    # and the last action overall must be "typing" (i.e. the override popped).
    first_record = actions.index("record_voice")
    last_record = len(actions) - 1 - list(reversed(actions)).index("record_voice")
    assert first_record > 0, "record_voice fired before any typing baseline tick"
    assert actions[-1] == "typing", (
        f"expected indicator to revert to typing after scope exit, "
        f"got actions[-1]={actions[-1]!r}"
    )
    assert last_record < len(actions) - 1, (
        "record_voice still firing after scope exited"
    )


# ---------------------------------------------------------------------------
# 4. Bonus: the override scope cleans up even when the body raises.
#    Guards against leaking the indicator on transcription errors.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_action_scope_pops_on_exception():
    """If the wrapped block raises (e.g. STT provider error), the action
    stack must still be popped — otherwise every subsequent message would
    show the microphone glyph forever.
    """
    adapter = _make_telegram_adapter()
    _attach_chat_action_recorder(adapter)

    with pytest.raises(RuntimeError, match="boom"):
        async with adapter._typing_action_scope("123", "record_voice"):
            assert adapter._typing_action_overrides["123"] == ["record_voice"]
            raise RuntimeError("boom")

    # Stack must be empty (and the chat key cleared) so the next refresh
    # tick falls back to the default action.
    assert "123" not in adapter._typing_action_overrides


# ---------------------------------------------------------------------------
# 5. Telegram adapter actually forwards ``action`` to send_chat_action.
#    (The previous tests cover the loop; this one nails down the wire.)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_send_typing_forwards_action_kwarg():
    """``TelegramAdapter.send_typing(action="record_voice")`` must pass the
    string straight through to ``Bot.send_chat_action`` — the upstream
    library forwards it to the Bot API as the ``action`` field.
    """
    adapter = _make_telegram_adapter()
    call_log = _attach_chat_action_recorder(adapter)

    await adapter.send_typing("456", action="record_voice")
    await adapter.send_typing("456")  # default

    assert call_log[0]["action"] == "record_voice"
    assert call_log[1]["action"] == "typing"
