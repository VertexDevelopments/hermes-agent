"""Tests for the ``pre_memory_write`` plugin hook.

This hook closes the bypass paths that ``pre_tool_call`` cannot cover:

* ``flush_memories()`` invokes ``tools.memory_tool.memory_tool`` directly
  during context compression, never going through the agent's main tool
  dispatcher (``run_agent.py:7502-7514``).
* External memory provider ``sync_all()`` writes after every turn without
  ``tool_name == "memory"`` (``run_agent.py:11897-11903``).

The corresponding ``maestro-memory-guard`` plugin hooks both bypasses to
enforce D-013 (project-confidentiality) and D-018 (Self Digest containment).
"""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.plugins as plugins_mod


def test_hook_registered():
    assert "pre_memory_write" in plugins_mod.VALID_HOOKS


def test_helper_returns_none_with_no_plugins():
    """No registered plugin → no block; the write proceeds."""
    with patch.object(plugins_mod, "invoke_hook", return_value=[]):
        result = plugins_mod.get_pre_memory_write_block_message(
            action="add",
            target="memory",
            content="hello",
            write_path="flush",
        )
    assert result is None


def test_helper_blocks_when_plugin_says_block():
    """A plugin returning ``{action: block, message: ...}`` blocks the write."""
    with patch.object(
        plugins_mod,
        "invoke_hook",
        return_value=[{"action": "block", "message": "no flushing here"}],
    ):
        result = plugins_mod.get_pre_memory_write_block_message(
            action="add",
            target="memory",
            content="anything",
            write_path="flush",
        )
    assert result == "no flushing here"


def test_helper_first_block_wins():
    """Multiple plugins; first non-empty block wins."""
    with patch.object(
        plugins_mod,
        "invoke_hook",
        return_value=[
            None,
            {"action": "allow"},
            {"action": "block", "message": "first block"},
            {"action": "block", "message": "second block (should not see)"},
        ],
    ):
        result = plugins_mod.get_pre_memory_write_block_message(
            action="add",
            target="memory",
            content="anything",
            write_path="provider_sync",
        )
    assert result == "first block"


def test_helper_ignores_non_block_returns():
    """Observer-only hooks (returning None or non-block dicts) don't break the
    write.  Mirrors ``get_pre_tool_call_block_message`` semantics."""
    with patch.object(
        plugins_mod,
        "invoke_hook",
        return_value=[None, "some string", 42, {"action": "allow"}],
    ):
        result = plugins_mod.get_pre_memory_write_block_message(
            action="replace",
            target="memory",
            content="x",
            old_text="y",
            write_path="tool",
        )
    assert result is None


def test_helper_empty_message_does_not_block():
    """A block dict with empty message is treated as non-block (mirrors
    pre_tool_call helper)."""
    with patch.object(
        plugins_mod,
        "invoke_hook",
        return_value=[{"action": "block", "message": ""}],
    ):
        result = plugins_mod.get_pre_memory_write_block_message(
            action="add",
            target="memory",
            content="anything",
            write_path="flush",
        )
    assert result is None


def test_helper_forwards_write_path():
    """Plugins receive the ``write_path`` so they can apply per-path policy."""
    captured = {}

    def fake_invoke(hook_name, **kwargs):
        captured["hook_name"] = hook_name
        captured["kwargs"] = kwargs
        return []

    with patch.object(plugins_mod, "invoke_hook", side_effect=fake_invoke):
        plugins_mod.get_pre_memory_write_block_message(
            action="add",
            target="memory",
            content="payload",
            old_text=None,
            write_path="provider_sync",
            session_id="sess-abc",
            skill_context={"active_skill": "maestro-zenflow", "project": "Zenflow"},
        )
    assert captured["hook_name"] == "pre_memory_write"
    assert captured["kwargs"]["write_path"] == "provider_sync"
    assert captured["kwargs"]["session_id"] == "sess-abc"
    assert captured["kwargs"]["skill_context"] == {
        "active_skill": "maestro-zenflow",
        "project": "Zenflow",
    }


def test_helper_skill_context_normalises_to_dict():
    """Non-dict skill_context arguments come through as an empty dict so
    plugins never have to defend against None."""
    captured = {}

    def fake_invoke(hook_name, **kwargs):
        captured.update(kwargs)
        return []

    with patch.object(plugins_mod, "invoke_hook", side_effect=fake_invoke):
        plugins_mod.get_pre_memory_write_block_message(
            action="add",
            target="memory",
            content="payload",
            write_path="flush",
            skill_context=None,  # type: ignore[arg-type]
        )
    assert captured["skill_context"] == {}
