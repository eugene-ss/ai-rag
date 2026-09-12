from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rag.embedding.base import Embedder
from rag.lexical.base import LexicalIndex
from rag.observability.tracing import span
from rag.schemas import Principal, ScoredChunk
from rag.security.acl import is_allowed
from rag.vectordb.base import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists with RRF: score = Σ w_i / (k + rank_i)."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        msg = "weights length must match ranked_lists"
        raise ValueError(msg)

    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for item in ranked:
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight * (1.0 / (k + item.rank))
            if cid not in best or item.score > best[cid].score:
                best[cid] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results: list[ScoredChunk] = []
    for rank, (cid, score) in enumerate(fused, start=1):
        src = best[cid]
        results.append(
            ScoredChunk(
                chunk=src.chunk,
                score=score,
                rank=rank,
                retriever="fused",
            )
        )
    return results


class HybridRetriever:
    """Dense + lexical retrieval with RRF fusion and ACL defense-in-depth."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        embedder: Embedder,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def retrieve(
        self,
        query: str,
        *,
        principal: Principal,
        top_k: int = 10,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        with span("hybrid_retrieve", top_k=top_k) as attrs:
            query_vec = self.embedder.embed_query(query)

            def dense() -> list[ScoredChunk]:
                return self.vector_store.search(
                    query_vec,
                    top_k=top_k,
                    principal=principal,
                    index_version=index_version,
                )

            def lexical() -> list[ScoredChunk]:
                return self.lexical_index.search(
                    query,
                    top_k=top_k,
                    principal=principal,
                    index_version=index_version,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_fut = pool.submit(dense)
                lex_fut = pool.submit(lexical)
                dense_hits = dense_fut.result()
                lex_hits = lex_fut.result()

            fused = reciprocal_rank_fusion(
                [dense_hits, lex_hits],
                k=self.rrf_k,
                weights=[self.dense_weight, self.lexical_weight],
            )
            # Defense in depth: re-check ACL after fusion.
            filtered = [s for s in fused if is_allowed(principal, s.chunk.acl)]
            # Re-rank positions after filter
            results = [
                ScoredChunk(
                    chunk=s.chunk,
                    score=s.score,
                    rank=i,
                    retriever="fused",
                )
                for i, s in enumerate(filtered[:top_k], start=1)
            ]
            attrs["dense"] = len(dense_hits)
            attrs["lexical"] = len(lex_hits)
            attrs["fused"] = len(results)
            return results
