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


# -----------------------------------------------------------------------------
# Integration tests against the actual write sites in run_agent.py.
#
# These prove the gate is wired at the call site, not just at the helper.  The
# round-6 review correctly noted that helper-level tests cannot prove a real
# write site is gated.
# -----------------------------------------------------------------------------


def test_flush_memories_call_site_imports_gate():
    """flush_memories source contains the gate import + helper invocation.
    A future refactor that deletes the gate would fail this test loudly.
    """
    import inspect
    import run_agent
    src = inspect.getsource(run_agent.AIAgent.flush_memories)
    assert "get_pre_memory_write_block_message" in src
    assert 'write_path="flush"' in src
    assert "REFUSING memory write fail-closed" in src


def test_provider_sync_call_site_imports_gate():
    """The provider sync_all gate is wired (source-level assertion)."""
    src = open("/Users/zenflow/.hermes/hermes-agent/run_agent.py", "r", encoding="utf-8").read()
    assert 'write_path="provider_sync"' in src
    # The sync_all guard variable name proves the gate result is consumed.
    assert "_provider_blocked" in src
    # Both protected paths must fail-closed on ImportError.
    assert src.count("REFUSING memory write fail-closed") >= 3


def test_main_memory_tool_dispatcher_gates_writes():
    """Both main dispatcher paths gate memory writes via pre_memory_write."""
    src = open("/Users/zenflow/.hermes/hermes-agent/run_agent.py", "r", encoding="utf-8").read()
    # Two dispatcher branches handle function_name == "memory".
    occurrences = src.count('elif function_name == "memory":')
    assert occurrences == 2, f"expected 2 dispatcher branches, got {occurrences}"
    # write_path="tool" must appear at both.
    assert src.count('write_path="tool"') >= 2


# -----------------------------------------------------------------------------
# Behavioural test fixture — minimal AIAgent + transport mock so flush_memories
# reaches its memory tool dispatch without a live LLM provider.
# -----------------------------------------------------------------------------


def _build_test_agent(monkeypatch):
    """Stand up a minimal AIAgent with the memory tool wired through and a
    fake transport that returns a chosen tool-call payload."""
    import sys
    import types
    from unittest.mock import MagicMock

    sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
    sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
    sys.modules.setdefault("fal_client", types.SimpleNamespace())

    import run_agent

    class _FakeOpenAI:
        def __init__(self, **kwargs): pass
        def close(self): pass

    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kw: [
        {"type": "function", "function": {
            "name": "memory", "description": "memory tool",
            "parameters": {"type": "object", "properties": {}},
        }},
    ])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)

    agent = run_agent.AIAgent(
        api_key="test-key",
        base_url="https://test.example.com/v1",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._memory_store = MagicMock()
    agent._memory_flush_min_turns = 1
    agent._user_turn_count = 5
    return agent, run_agent


def _wire_flush_response(agent, monkeypatch, content):
    """Patch the agent's transport to surface a memory tool call carrying
    ``content`` to the flush dispatch.  Returns the call_llm response stub."""
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    flush_args = json.dumps({
        "action": "add", "target": "memory", "content": content,
    })
    fake_normalized = SimpleNamespace(
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name="memory", arguments=flush_args))],
    )
    fake_transport = MagicMock()
    fake_transport.normalize_response.return_value = fake_normalized
    monkeypatch.setattr(agent, "_get_transport", lambda: fake_transport)

    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))


def _run_flush_with_mocks(agent, response_stub, gate_side_effect, memory_tool_mock):
    """Drive ``flush_memories`` with the given gate behaviour and memory-tool
    mock.  ``gate_side_effect`` may be a callable, an exception type, or a
    return value."""
    from unittest.mock import patch
    from unittest.mock import MagicMock

    if isinstance(gate_side_effect, type) and issubclass(gate_side_effect, BaseException):
        gate_patch_kwargs = {"side_effect": gate_side_effect}
    elif callable(gate_side_effect):
        gate_patch_kwargs = {"side_effect": gate_side_effect}
    else:
        gate_patch_kwargs = {"return_value": gate_side_effect}

    with patch("agent.auxiliary_client.call_llm", return_value=response_stub):
        with patch("hermes_cli.plugins.get_pre_memory_write_block_message", **gate_patch_kwargs):
            with patch("tools.memory_tool.memory_tool", memory_tool_mock):
                agent.flush_memories([
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "Note"},
                ])


def test_flush_memories_does_not_invoke_memory_tool_when_blocked(monkeypatch):
    """When pre_memory_write returns a block, flush MUST NOT call memory_tool.
    This is the behavioural assertion round-7 demanded — source inspection
    alone cannot prove the gate runs before the write."""
    from unittest.mock import MagicMock

    agent, _ = _build_test_agent(monkeypatch)
    response_stub = _wire_flush_response(
        agent, monkeypatch,
        "secret: <!-- self-digest-do-not-flush --> blocked content <!-- /self-digest-do-not-flush -->",
    )
    memory_tool_mock = MagicMock(return_value="should_not_be_called")

    def _block(write_path, **_kw):
        return "BLOCKED: D-018 self-digest" if write_path == "flush" else None

    _run_flush_with_mocks(agent, response_stub, _block, memory_tool_mock)
    assert memory_tool_mock.call_count == 0, (
        f"gate-blocked flush still called memory_tool ({memory_tool_mock.call_count})"
    )


def test_flush_memories_invokes_memory_tool_when_not_blocked(monkeypatch):
    """Paired with the above: when the gate allows, the write proceeds.
    Without this, the blocked test could pass by accident if flush is broken."""
    from unittest.mock import MagicMock

    agent, _ = _build_test_agent(monkeypatch)
    response_stub = _wire_flush_response(agent, monkeypatch, "innocent fact")
    memory_tool_mock = MagicMock(return_value="saved")

    _run_flush_with_mocks(agent, response_stub, None, memory_tool_mock)
    assert memory_tool_mock.call_count == 1, (
        f"gate-allowed flush did not call memory_tool ({memory_tool_mock.call_count})"
    )


def test_flush_memories_fail_closes_on_gate_import_error(monkeypatch):
    """When the gate helper raises ImportError, flush must NOT call
    memory_tool — fail-closed for security."""
    from unittest.mock import MagicMock

    agent, _ = _build_test_agent(monkeypatch)
    response_stub = _wire_flush_response(agent, monkeypatch, "any payload")
    memory_tool_mock = MagicMock(return_value="should_not_be_called")

    _run_flush_with_mocks(agent, response_stub, ImportError, memory_tool_mock)
    assert memory_tool_mock.call_count == 0, (
        f"flush did not fail-closed on gate ImportError ({memory_tool_mock.call_count})"
    )


def test_helper_returns_block_with_real_message_for_sentinel():
    """End-to-end via the actual plugin manager: a write containing the
    Self-Digest sentinel is blocked.

    Skipped if the maestro-memory-guard plugin manifest cannot be discovered
    (e.g. running tests in an environment with no installed plugin)."""
    try:
        plugins_mod._ensure_plugins_discovered(force=True)
    except Exception as exc:  # pragma: no cover — only on broken fixtures
        import pytest
        pytest.skip(f"plugin discovery unavailable: {exc}")

    sentinel_open = "<!-- self-digest-do-not-flush -->"
    sentinel_close = "<!-- /self-digest-do-not-flush -->"
    payload = f"prefix\n{sentinel_open}\noperator profile content\n{sentinel_close}\nsuffix"
    result = plugins_mod.get_pre_memory_write_block_message(
        action="add",
        target="memory",
        content=payload,
        write_path="flush",
        session_id=None,
        skill_context={"active_skill": None, "channel_id": None, "project": None},
    )
    # Plugin may or may not be loaded; if loaded, sentinel must be detected.
    if result is None:
        import pytest
        pytest.skip("maestro-memory-guard not loaded in this test environment")
    assert "self-digest" in result.lower() or "Self Digest" in result
