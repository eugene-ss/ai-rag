from __future__ import annotations

from rag.utils.hashing import sha256_hex, short_id
from rag.utils.text import collapse_blank_lines, normalize_whitespace, truncate


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("  a \n\t b  ") == "a b"


def test_collapse_blank_lines_preserves_paragraphs() -> None:
    assert collapse_blank_lines("a\n\n\n\nb") == "a\n\nb"
    assert collapse_blank_lines("a\n\nb") == "a\n\nb"


def test_truncate() -> None:
    assert truncate("abcdef", 10) == "abcdef"
    assert truncate("abcdef", 5) == "ab..."


def test_hashing_is_stable_and_field_separated() -> None:
    assert sha256_hex("a", "b") == sha256_hex("a", "b")
    # Separator prevents ("ab", "c") colliding with ("a", "bc")
    assert sha256_hex("ab", "c") != sha256_hex("a", "bc")
    assert len(short_id("x", length=24)) == 24
