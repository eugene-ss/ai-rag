from __future__ import annotations

from rag.generation.citations import resolve_citations
from rag.generation.refusal import should_refuse
from rag.schemas import AclTags, Chunk, Citation, ScoredChunk


def _scored(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=cid,
            doc_id="d",
            ordinal=0,
            text="alpha beta gamma",
            span=(0, 15),
            acl=AclTags(tenant="t", allow_groups=frozenset({"g"})),
            index_version="v1",
        ),
        score=score,
        rank=1,
        retriever="rerank",
    )


def test_should_refuse_low_score() -> None:
    refuse, reason = should_refuse(
        [_scored("abc12345deadbeef", 0.001)],
        [Citation(chunk_id="abc12345deadbeef", doc_id="d", quote="x")],
        score_threshold=0.01,
    )
    assert refuse
    assert reason == "low_confidence_score"


def test_should_refuse_zero_citations() -> None:
    refuse, reason = should_refuse(
        [_scored("abc12345deadbeef", 0.5)],
        [],
        score_threshold=0.01,
    )
    assert refuse
    assert reason == "zero_resolvable_citations"


def test_resolve_citations_from_markers() -> None:
    scored = [_scored("abc12345deadbeef", 0.9)]
    cites = resolve_citations("Answer [abc12345deadbeef] says so.", scored)
    assert len(cites) == 1
    assert cites[0].chunk_id == "abc12345deadbeef"
