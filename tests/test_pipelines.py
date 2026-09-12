from __future__ import annotations

from pathlib import Path

from rag.backends import BackendContext
from rag.index_registry import IndexRegistry
from rag.pipelines import offline as offline_mod
from rag.pipelines import online as online_mod
from rag.schemas import AclTags, Principal
from rag.settings import Settings
from tests.conftest import CORPUS


def _build_index(context: BackendContext, acl: AclTags, *, version: str = "v1") -> int:
    pipeline = offline_mod.from_context(context)
    pipeline.registry = IndexRegistry(context.settings.index_registry_path())
    return len(pipeline.run(CORPUS, index_version=version, acl=acl))


def test_offline_indexes_and_publishes_alias(context: BackendContext, acl: AclTags) -> None:
    count = _build_index(context, acl)
    assert count >= 3
    assert context.vector_store.resolve_alias("docs_live") == "v1"
    # Embedding model is stamped so a model swap invalidates the index.
    assert context.embedder.model_name

    registry = IndexRegistry(context.settings.index_registry_path())
    assert registry.active_version() == "v1"
    entry = registry.get("v1")
    assert entry is not None
    assert entry.status == "active"
    assert entry.chunk_count == count
    assert entry.embedding_model == context.embedder.model_name


def test_online_answers_with_citations(
    context: BackendContext, acl: AclTags, principal: Principal
) -> None:
    _build_index(context, acl)
    pipeline = online_mod.from_context(context)
    answer = pipeline.answer("What is hybrid retrieval?", principal=principal)

    assert answer.refused is False
    assert answer.citations
    assert answer.trace_id
    assert answer.index_version == "v1"
    assert answer.latency_ms > 0


def test_online_refuses_for_principal_outside_tenant(
    context: BackendContext, acl: AclTags, outsider: Principal
) -> None:
    _build_index(context, acl)
    pipeline = online_mod.from_context(context)
    answer = pipeline.answer("What is hybrid retrieval?", principal=outsider)

    assert answer.refused is True
    assert answer.refusal_reason == "no_retrieved_context"
    assert answer.citations == []


def test_refusal_on_empty_index(context: BackendContext, principal: Principal) -> None:
    pipeline = online_mod.from_context(context)
    answer = pipeline.answer("totally unknown topic xyzzy", principal=principal)
    assert answer.refused is True
    assert answer.refusal_reason == "no_retrieved_context"


def test_empty_query_is_refused(context: BackendContext, principal: Principal) -> None:
    pipeline = online_mod.from_context(context)
    answer = pipeline.answer("   ", principal=principal)
    assert answer.refused is True
    assert answer.refusal_reason == "empty_query"


def test_offline_accepts_relative_source_path(context: BackendContext, acl: AclTags) -> None:
    """File URIs require absolute paths; a relative source must still work."""
    relative = CORPUS.relative_to(Path.cwd()) if CORPUS.is_relative_to(Path.cwd()) else CORPUS
    pipeline = offline_mod.from_context(context)
    assert pipeline.run(relative, index_version="v1")


def test_build_without_activate_leaves_alias_untouched(
    context: BackendContext, acl: AclTags
) -> None:
    """Build-then-swap: a new version must not go live until promoted."""
    pipeline = offline_mod.from_context(context)
    pipeline.registry = IndexRegistry(context.settings.index_registry_path())

    pipeline.run(CORPUS, index_version="v1", acl=acl, activate=True)
    assert context.vector_store.resolve_alias("docs_live") == "v1"

    pipeline.run(CORPUS, index_version="v2", acl=acl, activate=False)
    assert context.vector_store.resolve_alias("docs_live") == "v1"

    pipeline.activate("v2")
    assert context.vector_store.resolve_alias("docs_live") == "v2"


def test_rollback_to_previous_version(context: BackendContext, acl: AclTags) -> None:
    pipeline = offline_mod.from_context(context)
    pipeline.registry = IndexRegistry(context.settings.index_registry_path())
    pipeline.run(CORPUS, index_version="v1", acl=acl, activate=True)
    pipeline.run(CORPUS, index_version="v2", acl=acl, activate=True)
    assert context.vector_store.resolve_alias("docs_live") == "v2"

    pipeline.activate("v1")
    assert context.vector_store.resolve_alias("docs_live") == "v1"
    assert IndexRegistry(context.settings.index_registry_path()).active_version() == "v1"


def test_malformed_document_does_not_abort_run(
    context: BackendContext, acl: AclTags, tmp_path: Path
) -> None:
    """A single unparseable file must not fail the whole reindex."""
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "good.md").write_text("# Title\n\nUseful content about retrieval.")
    (source / "mystery.bin").write_bytes(b"\x00\x01\x02binary")

    chunks = offline_mod.from_context(context).run(source, index_version="v1", acl=acl)
    assert chunks
    assert all(c.doc_id == "good" for c in chunks)


def test_multi_query_and_hyde_paths_retrieve(
    context: BackendContext, acl: AclTags, principal: Principal
) -> None:
    """multi_query and HyDE are wired into retrieval, not just importable."""
    _build_index(context, acl)
    tuned = context.settings.model_copy(update={"multi_query_enabled": True, "hyde_enabled": True})
    pipeline = online_mod.from_context(context)
    pipeline.settings = tuned

    result = pipeline.retrieve_only("What is hybrid retrieval?", principal=principal)
    assert len(result.rewritten) > 1
    assert result.results


def test_settings_reject_stub_backends_in_production() -> None:
    """Fail fast rather than serve production traffic from test stubs."""
    import pytest

    with pytest.raises(ValueError, match="Unsafe configuration"):
        Settings(env="production", auth_trust_headers=True)
