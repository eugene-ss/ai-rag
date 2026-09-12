from __future__ import annotations

from rag.prompts import get, list_prompts


def test_prompt_registry_loads_versioned_templates() -> None:
    names = {(n, v) for n, v in list_prompts()}
    assert ("answer_grounded", "v1") in names
    assert ("rewrite", "v1") in names
    assert ("hyde", "v1") in names


def test_answer_grounded_renders_required_placeholders() -> None:
    tmpl = get("answer_grounded", "v1")
    rendered = tmpl.render(
        question="What is RRF?",
        context="[chunk] Reciprocal Rank Fusion",
        prompt_version="v1",
    )
    assert "What is RRF?" in rendered
    assert "Reciprocal Rank Fusion" in rendered
    assert "v1" in rendered


def test_missing_placeholder_raises() -> None:
    tmpl = get("answer_grounded", "v1")
    try:
        tmpl.render(question="q", prompt_version="v1")
        raise AssertionError("expected KeyError")
    except KeyError as exc:
        assert "context" in str(exc)
