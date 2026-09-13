from __future__ import annotations

from rag.retrieval.hybrid import reciprocal_rank_fusion
from rag.schemas import AclTags, Chunk, ScoredChunk


def _scored(cid: str, rank: int, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id="d",
        ordinal=0,
        text=cid,
        span=(0, 1),
        acl=AclTags(tenant="t", allow_groups=frozenset({"g"})),
        index_version="v1",
    )
    return ScoredChunk(chunk=chunk, score=score, rank=rank, retriever="dense")


def test_rrf_prefers_items_appearing_in_both_lists() -> None:
    dense = [_scored("a", 1), _scored("b", 2), _scored("c", 3)]
    lexical = [_scored("b", 1), _scored("a", 2), _scored("d", 3)]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    top_ids = {s.chunk.chunk_id for s in fused[:2]}
    assert top_ids == {"a", "b"}
    assert fused[0].retriever == "fused"
    b = next(s for s in fused if s.chunk.chunk_id == "b")
    expected = 1 / 62 + 1 / 61
    assert abs(b.score - expected) < 1e-9


def test_rrf_weights() -> None:
    dense = [_scored("a", 1)]
    lexical = [_scored("b", 1)]
    fused = reciprocal_rank_fusion([dense, lexical], k=60, weights=[2.0, 0.0])
    assert fused[0].chunk.chunk_id == "a"
