"""SkillOpt v1 validation library tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


SKILL = """\
---
name: demo-skill
description: Demo skill for validation tests.
---

# Demo Skill

Keep REQUIRED_TOKEN in the guidance.
"""


def _write_skill(skill_dir: Path, content: str = SKILL) -> Path:
    skill_dir.mkdir(parents=True, exist_ok=True)
    target = skill_dir / "SKILL.md"
    target.write_text(content, encoding="utf-8")
    return target


def _write_suite(skill_dir: Path, suite: str) -> None:
    tests_dir = skill_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "suite.yaml").write_text(suite, encoding="utf-8")


def test_compute_patch_records_hashes_and_frontmatter_change():
    from tools.skill_patch_utils import compute_patch, frontmatter_changed

    patch = compute_patch(SKILL, "REQUIRED_TOKEN", "REQUIRED_TOKEN plus more")

    assert patch.error is None
    assert patch.match_count == 1
    assert patch.original_sha256 != patch.patched_sha256
    assert patch.patched_content != SKILL
    assert not frontmatter_changed(SKILL, patch.patched_content)


def test_validator_warns_without_suite_but_not_safety_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools.skill_patch_utils import compute_patch
    from tools.skill_validator import validate_skill_patch

    skill_dir = tmp_path / "skills" / "demo-skill"
    target = _write_skill(skill_dir)
    patch = compute_patch(SKILL, "REQUIRED_TOKEN", "REQUIRED_TOKEN plus more")

    result = validate_skill_patch(
        skill_name="demo-skill",
        skill_dir=skill_dir,
        target=target,
        patch=patch,
        mode="warn",
    )

    assert result.status == "no_test_suite"
    assert result.safety_failure is False


def test_validator_blocks_frontmatter_change_even_in_warn_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools.skill_patch_utils import compute_patch
    from tools.skill_validator import validate_skill_patch

    skill_dir = tmp_path / "skills" / "demo-skill"
    target = _write_skill(skill_dir)
    _write_suite(
        skill_dir,
        """
version: 1
test_cases:
  - name: token present
    validation_type: contains_all
    validation_params:
      values: [REQUIRED_TOKEN]
""",
    )
    patch = compute_patch(SKILL, "description: Demo skill", "description: Changed skill")

    result = validate_skill_patch(
        skill_name="demo-skill",
        skill_dir=skill_dir,
        target=target,
        patch=patch,
        mode="warn",
    )

    assert result.status == "rejected"
    assert result.safety_failure is True
    assert "frontmatter" in result.reason.lower()


def test_validator_rejects_suite_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools.skill_patch_utils import compute_patch
    from tools.skill_validator import validate_skill_patch

    skill_dir = tmp_path / "skills" / "demo-skill"
    target = _write_skill(skill_dir)
    outside = tmp_path / "outside-suite.yaml"
    outside.write_text("version: 1\ntest_cases: []\n", encoding="utf-8")
    tests_dir = skill_dir / "tests"
    tests_dir.mkdir(parents=True)
    suite = tests_dir / "suite.yaml"
    try:
        suite.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")

    patch = compute_patch(SKILL, "REQUIRED_TOKEN", "REQUIRED_TOKEN plus more")
    result = validate_skill_patch(
        skill_name="demo-skill",
        skill_dir=skill_dir,
        target=target,
        patch=patch,
        mode="warn",
    )

    assert result.status == "rejected"
    assert result.safety_failure is True
    assert "escape" in result.reason.lower() or "symlink" in result.reason.lower()


def test_regex_safe_uses_hard_timeout_for_catastrophic_pattern(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from tools.skill_patch_utils import compute_patch
    from tools.skill_validator import validate_skill_patch

    content = SKILL + "\n" + ("a" * 50_000) + "!\n"
    skill_dir = tmp_path / "skills" / "demo-skill"
    target = _write_skill(skill_dir, content)
    _write_suite(
        skill_dir,
        """
version: 1
test_cases:
  - name: catastrophic regex must not hang parent process
    validation_type: regex_safe
    validation_params:
      pattern: "(a+)+$"
""",
    )
    patch = compute_patch(content, "REQUIRED_TOKEN", "REQUIRED_TOKEN plus more")

    start = time.monotonic()
    result = validate_skill_patch(
        skill_name="demo-skill",
        skill_dir=skill_dir,
        target=target,
        patch=patch,
        mode="warn",
        config={"regex_timeout_ms": 10, "regex_max_content_chars": 100_000},
    )
    elapsed = time.monotonic() - start

    assert elapsed < 5
    assert result.status == "rejected"
    assert result.safety_failure is True
    assert "regex" in result.reason.lower() or "timeout" in result.reason.lower()


def test_pending_cleanup_enforces_record_and_byte_caps(tmp_path):
    from tools.skill_sidecar_utils import cleanup_pending

    pending = tmp_path / ".pending"
    pending.mkdir()
    for idx in range(10):
        p = pending / f"{idx:02d}.json"
        p.write_text(json.dumps({"idx": idx, "payload": "x" * 200}), encoding="utf-8")
        os.utime(p, (time.time() + idx, time.time() + idx))

    cleanup_pending(pending, ttl_days=30, max_records=3, max_bytes=2_000)

    remaining = sorted(p.name for p in pending.glob("*.json"))
    assert len(remaining) <= 3
    assert remaining == ["07.json", "08.json", "09.json"]


def test_pending_cleanup_rejects_pending_root_symlink_escape(tmp_path):
    from tools.skill_sidecar_utils import SidecarPathError, cleanup_pending

    skills_dir = tmp_path / "skills"
    outside = tmp_path / "outside"
    skills_dir.mkdir()
    outside.mkdir()
    pending = skills_dir / ".pending"
    try:
        pending.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not supported")

    with pytest.raises(SidecarPathError):
        cleanup_pending(pending, ttl_days=30, max_records=3, max_bytes=2_000)
