from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip."""
    return _WHITESPACE.sub(" ", text).strip()


def collapse_blank_lines(text: str) -> str:
    """Collapse 3+ newlines to a paragraph break, preserving structure."""
    return _BLANK_LINES.sub("\n\n", text.strip())


def truncate(text: str, limit: int, *, suffix: str = "...") -> str:
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix
