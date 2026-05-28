"""Integration tests for opt-in skill patch validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.registry import registry
from tools.skill_manager_tool import _create_skill, skill_manage


VALID_SKILL_CONTENT = """\
---
name: demo-skill
description: Demo skill for validation integration tests.
---

# Demo Skill

Keep REQUIRED_TOKEN in the guidance.
"""


def _skill_dir(tmp_path):
    return patch.multiple(
        "tools.skill_manager_tool",
        SKILLS_DIR=tmp_path,
    )


class _SkillDirContext:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self._patches = []

    def __enter__(self):
        self._patches = [
            patch("tools.skill_manager_tool.SKILLS_DIR", self.tmp_path),
            patch("agent.skill_utils.get_all_skills_dirs", return_value=[self.tmp_path]),
        ]
        for p in self._patches:
            p.__enter__()
        return self.tmp_path

    def __exit__(self, exc_type, exc, tb):
        for p in reversed(self._patches):
            p.__exit__(exc_type, exc, tb)


def _isolated_skill_dir(tmp_path):
    return _SkillDirContext(tmp_path)


def _write_suite(skill_dir: Path, required_value: str) -> None:
    tests_dir = skill_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "suite.yaml").write_text(
        f"""
version: 1
threshold: 1.0
test_cases:
  - name: required value
    validation_type: contains_all
    validation_params:
      values: [{required_value}]
""",
        encoding="utf-8",
    )


def test_validate_false_preserves_legacy_patch_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    with _isolated_skill_dir(tmp_path):
        _create_skill("demo-skill", VALID_SKILL_CONTENT)
        raw = skill_manage(
            action="patch",
            name="demo-skill",
            old_string="REQUIRED_TOKEN",
            new_string="REQUIRED_TOKEN plus more",
        )

    result = json.loads(raw)
    assert result["success"] is True
    assert "validation" not in result
    assert "REQUIRED_TOKEN plus more" in (tmp_path / "demo-skill" / "SKILL.md").read_text()


def test_validation_warn_mode_applies_patch_and_returns_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    with _isolated_skill_dir(tmp_path):
        _create_skill("demo-skill", VALID_SKILL_CONTENT)
        _write_suite(tmp_path / "demo-skill", "MISSING_TOKEN")
        raw = skill_manage(
            action="patch",
            name="demo-skill",
            old_string="REQUIRED_TOKEN",
            new_string="REQUIRED_TOKEN plus more",
            validate=True,
            validation_mode="warn",
        )

    result = json.loads(raw)
    assert result["success"] is True
    assert result["validation"]["status"] == "warning"
    assert "REQUIRED_TOKEN plus more" in (tmp_path / "demo-skill" / "SKILL.md").read_text()


def test_validation_blocking_mode_rejects_and_preserves_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    with _isolated_skill_dir(tmp_path):
        _create_skill("demo-skill", VALID_SKILL_CONTENT)
        _write_suite(tmp_path / "demo-skill", "MISSING_TOKEN")
        raw = skill_manage(
            action="patch",
            name="demo-skill",
            old_string="REQUIRED_TOKEN",
            new_string="REQUIRED_TOKEN plus more",
            validate=True,
            validation_mode="blocking",
        )

    result = json.loads(raw)
    assert result["success"] is False
    assert result["validation"]["status"] == "rejected"
    assert "REQUIRED_TOKEN plus more" not in (tmp_path / "demo-skill" / "SKILL.md").read_text()


def test_registry_schema_and_handler_forward_validation_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    entry = registry.get_entry("skill_manage")
    assert entry is not None
    props = entry.schema["parameters"]["properties"]
    assert "validate" in props
    assert "validation_mode" in props

    with _isolated_skill_dir(tmp_path):
        _create_skill("demo-skill", VALID_SKILL_CONTENT)
        _write_suite(tmp_path / "demo-skill", "MISSING_TOKEN")
        raw = entry.handler(
            {
                "action": "patch",
                "name": "demo-skill",
                "old_string": "REQUIRED_TOKEN",
                "new_string": "REQUIRED_TOKEN plus more",
                "validate": True,
                "validation_mode": "blocking",
            }
        )

    result = json.loads(raw)
    assert result["success"] is False
    assert result["validation"]["status"] == "rejected"


def test_validation_rejects_if_target_changes_after_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools import skill_validator

    original_validate = skill_validator.validate_skill_patch

    def mutate_target_then_validate(**kwargs):
        kwargs["target"].write_text(kwargs["target"].read_text() + "\nconcurrent change\n")
        return original_validate(**kwargs)

    with _isolated_skill_dir(tmp_path):
        _create_skill("demo-skill", VALID_SKILL_CONTENT)
        _write_suite(tmp_path / "demo-skill", "REQUIRED_TOKEN plus more")
        with patch("tools.skill_validator.validate_skill_patch", side_effect=mutate_target_then_validate):
            raw = skill_manage(
                action="patch",
                name="demo-skill",
                old_string="REQUIRED_TOKEN",
                new_string="REQUIRED_TOKEN plus more",
                validate=True,
                validation_mode="blocking",
            )

    result = json.loads(raw)
    assert result["success"] is False
    assert result["validation"]["status"] == "rejected"
    assert "changed after validation" in result["error"].lower()
