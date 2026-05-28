"""Symlink-safe sidecar helpers for SkillOpt validation state."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
from typing import Any

from utils import atomic_json_write


class SidecarPathError(ValueError):
    """Raised when a validation sidecar path escapes its allowed root."""


def _parts_have_traversal(parts: tuple[str, ...]) -> bool:
    return any(part in {"..", ""} for part in parts)


def safe_child_path(root: Path, *parts: str) -> Path:
    """Return ``root / parts`` after verifying it stays under ``root``.

    Absolute path components and ``..`` traversal components are rejected before
    resolution. Existing symlinks in parent directories are resolved and must
    still remain inside the resolved root.
    """

    root = Path(root)
    candidate = root
    for raw in parts:
        part = Path(str(raw))
        if part.is_absolute() or _parts_have_traversal(part.parts):
            raise SidecarPathError(f"Path escapes allowed directory: {raw}")
        candidate = candidate / part

    root_resolved = root.resolve()
    try:
        candidate_resolved = candidate.resolve(strict=False)
        candidate_resolved.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SidecarPathError(f"Path escapes allowed directory: {exc}") from exc
    return candidate


def reject_symlink_escape(path: Path, root: Path) -> None:
    """Reject existing symlinks/parents that resolve outside ``root``."""

    root = Path(root)
    path = Path(path)
    root_resolved = root.resolve()
    try:
        # ``resolve(strict=False)`` follows all existing symlink components while
        # permitting a new leaf file. That catches parent symlink escapes and an
        # existing symlink leaf pointing outside the root.
        path.resolve(strict=False).relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SidecarPathError(f"Path escapes allowed directory: {exc}") from exc


def reject_root_symlink_escape(root: Path) -> None:
    """Reject a sidecar root that is itself a symlink escaping its parent.

    ``reject_symlink_escape(root / 'file', root)`` is not enough when ``root``
    already resolves outside the profile/skills directory, because the resolved
    file is still relative to the resolved root. Pending roots therefore need a
    parent-anchored check before any write.
    """

    root = Path(root)
    if not root.exists() and not root.is_symlink():
        return
    try:
        root.resolve(strict=False).relative_to(root.parent.resolve(strict=False))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SidecarPathError(f"Sidecar root escapes allowed directory: {exc}") from exc


def atomic_write_json_safe(path: Path, data: Any, root: Path) -> None:
    """Atomically write JSON after symlink-escape validation."""

    reject_symlink_escape(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_escape(path.parent, root)
    atomic_json_write(path, data)


def _parse_expiry(path: Path) -> datetime | None:
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        expires = raw.get("expires_at")
        if isinstance(expires, str) and expires:
            if expires.endswith("Z"):
                expires = expires[:-1] + "+00:00"
            return datetime.fromisoformat(expires)
    except Exception:
        return None
    return None


def cleanup_pending(
    root: Path,
    ttl_days: int,
    max_records: int,
    max_bytes: int,
) -> None:
    """Bound pending validation records by age, count, and total bytes."""

    root = Path(root)
    reject_root_symlink_escape(root)
    root.mkdir(parents=True, exist_ok=True)
    reject_root_symlink_escape(root)
    reject_symlink_escape(root, root)

    now = datetime.now(timezone.utc)
    ttl_cutoff = now - timedelta(days=max(0, int(ttl_days)))

    records = [p for p in root.glob("*.json") if p.is_file() or p.is_symlink()]
    for record in list(records):
        try:
            reject_symlink_escape(record, root)
            expiry = _parse_expiry(record)
            mtime = datetime.fromtimestamp(record.stat().st_mtime, timezone.utc)
            if (expiry is not None and expiry < now) or mtime < ttl_cutoff:
                record.unlink(missing_ok=True)
        except Exception:
            # Broken/unsafe pending files do not deserve to survive cleanup.
            try:
                record.unlink(missing_ok=True)
            except OSError:
                pass

    records = [p for p in root.glob("*.json") if p.is_file()]
    records.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if max_records >= 0:
        for record in records[max_records:]:
            record.unlink(missing_ok=True)
        records = records[:max_records]

    if max_bytes >= 0:
        total = sum(p.stat().st_size for p in records if p.exists())
        for record in sorted(records, key=lambda p: p.stat().st_mtime):
            if total <= max_bytes:
                break
            try:
                size = record.stat().st_size
                record.unlink(missing_ok=True)
                total -= size
            except OSError:
                pass
