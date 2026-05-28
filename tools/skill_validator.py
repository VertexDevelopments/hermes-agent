"""Opt-in SkillOpt-inspired validation for skill patches.

This is a library module, not a tool. It deliberately avoids importing
``tools.registry`` or ``tools.skill_manager_tool`` so it can be used by the
skill manager without circular registration side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Optional

import yaml

from hermes_constants import get_config_path, get_hermes_home
from tools.path_security import validate_within_dir
from tools.skill_patch_utils import PatchComputation, frontmatter_changed
from tools.skill_sidecar_utils import (
    SidecarPathError,
    atomic_write_json_safe,
    cleanup_pending,
    reject_symlink_escape,
    safe_child_path,
)
from utils import is_truthy_value


VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MAX_NAME_LENGTH = 64

DEFAULT_VALIDATION_CONFIG: dict[str, Any] = {
    "enabled": True,
    "default_validate": False,
    "default_mode": "warn",
    "fail_open": False,
    "allow_no_suite": True,
    "threshold": 0.8,
    "max_chars_per_patch": 500,
    "max_patches_per_skill_per_day": 10,
    "pending_ttl_days": 7,
    "max_pending_records": 500,
    "max_pending_bytes": 1_048_576,
    "suite_max_bytes": 65_536,
    "max_test_cases": 50,
    "protect_yaml_frontmatter": True,
    "regex_max_pattern_chars": 300,
    "regex_max_content_chars": 100_000,
    "regex_timeout_ms": 250,
}


@dataclass
class ValidationResult:
    status: str
    mode: str
    score: float
    safety_failure: bool
    test_results: list[dict[str, Any]]
    failed_tests: list[str]
    warnings: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validation_disabled_by_env() -> bool:
    value = os.getenv("HERMES_SKILL_VALIDATION")
    if value is None:
        return False
    return value.strip().lower() in {"0", "false", "off", "no"}


def _merge_config(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(DEFAULT_VALIDATION_CONFIG)
    if config:
        merged.update(config)
    return merged


def load_validation_config() -> dict[str, Any]:
    """Load skill validation config without importing CLI entrypoints."""

    cfg = dict(DEFAULT_VALIDATION_CONFIG)
    path = get_config_path()
    if not path.exists():
        return cfg
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return cfg
    if not isinstance(parsed, dict):
        return cfg
    section = parsed.get("skill_optimization")
    if not isinstance(section, dict):
        return cfg
    validation = section.get("validation")
    if isinstance(validation, dict):
        # ``auto_accept`` was used by a rejected draft. Ignore it; it has no
        # meaning in the final API.
        validation = {k: v for k, v in validation.items() if k != "auto_accept"}
        cfg.update(validation)
    return cfg


def _result(
    *,
    status: str,
    mode: str,
    reason: str,
    score: float = 0.0,
    safety_failure: bool = False,
    test_results: Optional[list[dict[str, Any]]] = None,
    failed_tests: Optional[list[str]] = None,
    warnings: Optional[list[str]] = None,
) -> ValidationResult:
    return ValidationResult(
        status=status,
        mode=mode,
        score=score,
        safety_failure=safety_failure,
        test_results=test_results or [],
        failed_tests=failed_tests or [],
        warnings=warnings or [],
        reason=reason,
    )


def _validate_skill_name(skill_name: str) -> Optional[str]:
    if not skill_name:
        return "Skill name is required."
    if len(skill_name) > MAX_NAME_LENGTH:
        return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
    if not VALID_NAME_RE.match(skill_name):
        return f"Invalid skill name '{skill_name}'."
    return None


def _target_relpath(skill_dir: Path, target: Path) -> str:
    try:
        return str(target.resolve(strict=False).relative_to(skill_dir.resolve()))
    except Exception:
        return target.name


def _pending_root() -> Path:
    return get_hermes_home() / "skills" / ".pending"


def _patch_key(skill_name: str, target_relpath: str, patch: PatchComputation) -> str:
    raw = f"{skill_name}\0{target_relpath}\0{patch.original_sha256}\0{patch.patched_sha256}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pending_records_for_today(root: Path, skill_name: str) -> int:
    today = _now().date().isoformat()
    count = 0
    for record in root.glob("*.json"):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except Exception:
            continue
        created = str(data.get("created_at", ""))[:10]
        if data.get("skill_name") == skill_name and created == today:
            count += 1
    return count


def _write_pending_record(
    *,
    skill_name: str,
    skill_dir: Path,
    target: Path,
    patch: PatchComputation,
    result: ValidationResult,
    cfg: dict[str, Any],
) -> None:
    root = _pending_root()
    cleanup_pending(
        root,
        ttl_days=int(cfg.get("pending_ttl_days", 7)),
        max_records=int(cfg.get("max_pending_records", 500)),
        max_bytes=int(cfg.get("max_pending_bytes", 1_048_576)),
    )
    target_relpath = _target_relpath(skill_dir, target)
    key = _patch_key(skill_name, target_relpath, patch)
    created = _now()
    record = {
        "patch_hash": key,
        "skill_name": skill_name,
        "skill_dir": str(skill_dir.resolve()),
        "target_relpath": target_relpath,
        "validated_content_sha256": patch.patched_sha256,
        "status": result.status,
        "score": result.score,
        "created_at": _iso(created),
        "expires_at": _iso(created + timedelta(days=int(cfg.get("pending_ttl_days", 7)))),
    }
    atomic_write_json_safe(root / f"{key}.json", record, root)


def _redacted_preview(text: str, limit: int = 200) -> str:
    preview = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+", r"\1=<redacted>", text)
    preview = preview.replace("\n", "\\n")
    return preview[:limit]


def _append_rejected_edit(
    *,
    skill_name: str,
    skill_dir: Path,
    target: Path,
    patch: PatchComputation,
    result: ValidationResult,
) -> None:
    path = safe_child_path(skill_dir, ".rejected_edits.json")
    try:
        reject_symlink_escape(path, skill_dir)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        existing = {}
    records = existing.get("rejected_edits") if isinstance(existing, dict) else []
    if not isinstance(records, list):
        records = []
    target_relpath = _target_relpath(skill_dir, target)
    records.append(
        {
            "patch_hash": _patch_key(skill_name, target_relpath, patch),
            "timestamp": _iso(_now()),
            "reason": result.reason,
            "score": result.score,
            "failed_tests": result.failed_tests,
            "target_relpath": target_relpath,
            "patch_preview": _redacted_preview(patch.patched_content),
        }
    )
    payload = {"skill_name": skill_name, "rejected_edits": records[-50:]}
    atomic_write_json_safe(path, payload, skill_dir)


def _suite_path(skill_dir: Path) -> Path:
    return safe_child_path(skill_dir, "tests", "suite.yaml")


def _load_suite(skill_dir: Path, cfg: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        suite = _suite_path(skill_dir)
        if not suite.exists():
            return None, None
        reject_symlink_escape(suite, skill_dir)
        size = suite.stat().st_size
        max_bytes = int(cfg.get("suite_max_bytes", 65_536))
        if size > max_bytes:
            return None, f"Test suite is {size:,} bytes (limit: {max_bytes:,})."
        parsed = yaml.safe_load(suite.read_text(encoding="utf-8"))
    except (OSError, SidecarPathError, yaml.YAMLError) as exc:
        return None, f"Unsafe or malformed test suite: {exc}"
    if parsed is None:
        return {"version": 1, "test_cases": []}, None
    if not isinstance(parsed, dict):
        return None, "Test suite must be a YAML mapping."
    cases = parsed.get("test_cases", [])
    if not isinstance(cases, list):
        return None, "test_cases must be a list."
    max_cases = int(cfg.get("max_test_cases", 50))
    if len(cases) > max_cases:
        return None, f"Test suite has {len(cases)} test cases (limit: {max_cases})."
    return parsed, None


def _literal_values(params: Any) -> list[str]:
    if not isinstance(params, dict):
        return []
    values = params.get("values")
    if values is None and "value" in params:
        values = [params.get("value")]
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def _run_regex_safe(pattern: str, content: str, params: dict[str, Any], cfg: dict[str, Any]) -> tuple[bool, Optional[str]]:
    max_pattern = int(cfg.get("regex_max_pattern_chars", 300))
    max_content = int(cfg.get("regex_max_content_chars", 100_000))
    if len(pattern) > max_pattern:
        return False, f"regex pattern exceeds {max_pattern} chars"
    if len(content) > max_content:
        return False, f"regex target content exceeds {max_content} chars"

    timeout_s = max(0.001, int(cfg.get("regex_timeout_ms", 250)) / 1000.0)
    child_code = r'''
import json, re, sys
payload = json.loads(sys.stdin.read())
flags = re.IGNORECASE if payload.get("ignore_case") else 0
try:
    matched = re.search(payload["pattern"], payload["content"], flags) is not None
    print(json.dumps({"matched": matched}))
except re.error as exc:
    print(json.dumps({"error": str(exc)}))
'''
    payload = {
        "pattern": pattern,
        "content": content,
        "ignore_case": bool(params.get("ignore_case", False)),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", child_code],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "regex timed out"
    if proc.returncode != 0:
        return False, f"regex subprocess failed: {proc.stderr[:200]}"
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, "regex subprocess returned malformed output"
    if data.get("error"):
        return False, f"regex error: {data['error']}"
    return bool(data.get("matched")), None


def _frontmatter_yaml_ok(content: str) -> tuple[bool, Optional[str]]:
    from tools.skill_patch_utils import frontmatter_block

    block = frontmatter_block(content)
    if block is None:
        return False, "missing or unclosed frontmatter"
    yaml_text = block[3:]
    end = yaml_text.rfind("---")
    if end >= 0:
        yaml_text = yaml_text[:end]
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return False, f"frontmatter YAML parse error: {exc}"
    if not isinstance(parsed, dict):
        return False, "frontmatter must be a mapping"
    return True, None


def _run_test_case(case: Any, content: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(case, dict):
        return {"name": "<invalid>", "success": False, "error": "test case must be a mapping", "safety_failure": True}
    name = str(case.get("name") or "<unnamed>")
    validation_type = str(case.get("validation_type") or "").strip()
    params = case.get("validation_params") or {}
    if not isinstance(params, dict):
        params = {}

    if validation_type == "contains_all":
        values = _literal_values(params)
        missing = [value for value in values if value not in content]
        return {"name": name, "success": not missing, "missing": missing}
    if validation_type == "contains_any":
        values = _literal_values(params)
        matched = any(value in content for value in values)
        return {"name": name, "success": matched, "missing_any_of": values if not matched else []}
    if validation_type == "contains_none":
        values = _literal_values(params)
        present = [value for value in values if value in content]
        return {"name": name, "success": not present, "present": present}
    if validation_type == "frontmatter_yaml":
        ok, err = _frontmatter_yaml_ok(content)
        return {"name": name, "success": ok, "error": err}
    if validation_type == "regex_safe":
        pattern = str(params.get("pattern") or "")
        ok, err = _run_regex_safe(pattern, content, params, cfg)
        result = {"name": name, "success": ok}
        if err:
            result["error"] = err
            result["safety_failure"] = True
        return result

    return {
        "name": name,
        "success": False,
        "error": f"unsupported validation_type '{validation_type}'",
        "safety_failure": True,
    }


def validate_skill_patch(
    *,
    skill_name: str,
    skill_dir: Path,
    target: Path,
    patch: PatchComputation,
    mode: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> ValidationResult:
    """Validate a precomputed skill patch.

    The caller owns writing. This function returns metadata only and never
    writes the patched skill content.
    """

    cfg = _merge_config(load_validation_config())
    cfg.update(config or {})
    mode = str(mode or cfg.get("default_mode") or "warn").strip().lower()
    if mode not in {"warn", "blocking"}:
        mode = "warn"

    if _validation_disabled_by_env() or not is_truthy_value(cfg.get("enabled"), default=True):
        return _result(status="skipped_by_env", mode=mode, reason="skill validation disabled")

    name_error = _validate_skill_name(skill_name)
    if name_error:
        return _result(status="rejected", mode=mode, reason=name_error, safety_failure=True)

    skill_dir = Path(skill_dir)
    target = Path(target)
    path_error = validate_within_dir(target, skill_dir)
    if path_error:
        return _result(status="rejected", mode=mode, reason=path_error, safety_failure=True)

    if bool(cfg.get("protect_yaml_frontmatter", True)) and target.name == "SKILL.md":
        try:
            original_content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return _result(status="rejected", mode=mode, reason=f"could not read target: {exc}", safety_failure=True)
        if frontmatter_changed(original_content, patch.patched_content):
            result = _result(
                status="rejected",
                mode=mode,
                reason="patch touches protected YAML frontmatter",
                safety_failure=True,
            )
            _append_rejected_edit(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result)
            _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
            return result

    delta_chars = abs(len(patch.patched_content) - len(target.read_text(encoding="utf-8"))) if target.exists() else 0
    max_delta = int(cfg.get("max_chars_per_patch", 500))
    if delta_chars > max_delta:
        result = _result(
            status="rejected",
            mode=mode,
            reason=f"patch changes {delta_chars} chars (limit: {max_delta})",
            safety_failure=True,
        )
        _append_rejected_edit(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result)
        _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
        return result

    pending = _pending_root()
    cleanup_pending(
        pending,
        ttl_days=int(cfg.get("pending_ttl_days", 7)),
        max_records=int(cfg.get("max_pending_records", 500)),
        max_bytes=int(cfg.get("max_pending_bytes", 1_048_576)),
    )
    max_daily = int(cfg.get("max_patches_per_skill_per_day", 10))
    if max_daily >= 0 and _pending_records_for_today(pending, skill_name) >= max_daily:
        return _result(
            status="rejected",
            mode=mode,
            reason=f"daily validation cap reached for skill '{skill_name}'",
            safety_failure=True,
        )

    suite, suite_error = _load_suite(skill_dir, cfg)
    if suite_error:
        result = _result(status="rejected", mode=mode, reason=suite_error, safety_failure=True)
        _append_rejected_edit(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result)
        _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
        return result

    if suite is None:
        allow = bool(cfg.get("allow_no_suite", True)) or mode == "warn"
        result = _result(
            status="no_test_suite" if allow else "rejected",
            mode=mode,
            reason="no validation test suite found",
            safety_failure=not allow,
            warnings=["no validation test suite found"] if allow else [],
        )
        if result.status == "rejected":
            _append_rejected_edit(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result)
        _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
        return result

    suite_mode = suite.get("mode") if isinstance(suite.get("mode"), str) else None
    if mode is None and suite_mode in {"warn", "blocking"}:
        mode = suite_mode
    threshold = float(suite.get("threshold", cfg.get("threshold", 0.8)))
    cases = suite.get("test_cases") or []
    test_results = [_run_test_case(case, patch.patched_content, cfg) for case in cases]
    safety_failure = any(bool(item.get("safety_failure")) for item in test_results)
    passed = sum(1 for item in test_results if item.get("success"))
    score = 1.0 if not test_results else passed / len(test_results)
    failed = [str(item.get("name") or "<unnamed>") for item in test_results if not item.get("success")]

    if safety_failure:
        safety_reasons = [str(item.get("error")) for item in test_results if item.get("safety_failure") and item.get("error")]
        reason = safety_reasons[0] if safety_reasons else "validation safety failure"
        result = _result(
            status="rejected",
            mode=mode,
            score=score,
            safety_failure=True,
            test_results=test_results,
            failed_tests=failed,
            reason=reason,
        )
        _append_rejected_edit(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result)
        _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
        return result

    if score < threshold:
        status = "warning" if mode == "warn" else "rejected"
        result = _result(
            status=status,
            mode=mode,
            score=score,
            safety_failure=False,
            test_results=test_results,
            failed_tests=failed,
            warnings=["validation score below threshold"] if status == "warning" else [],
            reason=f"validation score {score:.2f} below threshold {threshold:.2f}",
        )
        if status == "rejected":
            _append_rejected_edit(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result)
        _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
        return result

    result = _result(
        status="accepted",
        mode=mode,
        score=score,
        safety_failure=False,
        test_results=test_results,
        failed_tests=[],
        reason="validation passed",
    )
    _write_pending_record(skill_name=skill_name, skill_dir=skill_dir, target=target, patch=patch, result=result, cfg=cfg)
    return result
