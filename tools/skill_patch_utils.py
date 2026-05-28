"""Registry-free helpers for computing skill patch content.

This module intentionally does not import ``tools.registry`` or
``tools.skill_manager_tool``.  It is safe for both the skill manager tool and
validation library to import without triggering tool registration side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Optional


@dataclass(frozen=True)
class PatchComputation:
    """Result of computing a skill patch against one original content blob."""

    original_sha256: str
    patched_sha256: str
    patched_content: str
    match_count: int
    strategy: Optional[str]
    error: Optional[str]
    match_spans: list[tuple[int, int]]


def sha256_text(content: str) -> str:
    """Return the SHA-256 hex digest for UTF-8 text."""

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _exact_spans(content: str, old_string: str, replace_all: bool) -> list[tuple[int, int]]:
    if not old_string:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        pos = content.find(old_string, start)
        if pos == -1:
            break
        spans.append((pos, pos + len(old_string)))
        start = pos + 1
    if not replace_all and len(spans) != 1:
        return []
    return spans


def compute_patch(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> PatchComputation:
    """Compute a patch once using Hermes' fuzzy replacement engine.

    The caller must write exactly ``patched_content`` after validating it and
    after re-checking that the on-disk original still has ``original_sha256``.
    """

    from tools.fuzzy_match import fuzzy_find_and_replace

    patched_content, match_count, strategy, error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all
    )
    spans = _exact_spans(content, old_string, replace_all) if not error and strategy == "exact" else []
    return PatchComputation(
        original_sha256=sha256_text(content),
        patched_sha256=sha256_text(patched_content),
        patched_content=patched_content,
        match_count=match_count,
        strategy=strategy,
        error=error,
        match_spans=spans,
    )


def frontmatter_block(content: str) -> Optional[str]:
    """Return the full YAML frontmatter block, including delimiters.

    Returns ``None`` when the text has no closed frontmatter block.
    """

    if not content.startswith("---"):
        return None
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return None
    return content[: end_match.end() + 3]


def frontmatter_changed(original: str, patched: str) -> bool:
    """True when the frontmatter block changed between two content blobs."""

    return frontmatter_block(original) != frontmatter_block(patched)
