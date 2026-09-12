from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rag.embedding.base import Embedder
from rag.lexical.base import LexicalIndex
from rag.observability.tracing import span
from rag.retrieval.filters import AclFilter
from rag.schemas import Principal, ScoredChunk
from rag.security.acl import is_allowed
from rag.vectordb.base import VectorStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists with RRF: score = Σ w_i / (k + rank_i).

    Rank-based fusion avoids comparing a cosine similarity against a BM25 score,
    which are on incomparable scales.
    """
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
        max_workers: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.max_workers = max_workers

    def retrieve(
        self,
        query: str,
        *,
        principal: Principal,
        top_k: int = 10,
        index_version: str | None = None,
        dense_query: str | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve for a single query string.

        `dense_query` overrides the text used for the dense leg only, which is
        how HyDE works: embed a hypothetical answer, keep BM25 on the real words.
        """
        return self.retrieve_multi(
            [query],
            principal=principal,
            top_k=top_k,
            index_version=index_version,
            dense_queries=[dense_query or query],
        )

    def retrieve_multi(
        self,
        queries: list[str],
        *,
        principal: Principal,
        top_k: int = 10,
        index_version: str | None = None,
        dense_queries: list[str] | None = None,
    ) -> list[ScoredChunk]:
        """Fan out over query variants, then fuse every ranked list at once.

        Multi-query retrieval and hybrid retrieval are the same operation here:
        N dense legs plus N lexical legs, all fused with RRF.
        """
        if not queries:
            return []
        dense_texts = dense_queries or list(queries)
        if len(dense_texts) != len(queries):
            msg = "dense_queries length must match queries"
            raise ValueError(msg)

        acl = AclFilter.for_principal(principal, index_version=index_version)
        with span(
            "hybrid_retrieve",
            top_k=top_k,
            variants=len(queries),
            # Fingerprint, never the identity: enough to explain an empty result
            # set without writing subject or group names into traces.
            acl=acl.fingerprint,
        ) as attrs:
            vectors = self.embedder.embed(dense_texts)

            def dense(vector: list[float]) -> list[ScoredChunk]:
                return self.vector_store.search(
                    vector,
                    top_k=top_k,
                    principal=principal,
                    index_version=index_version,
                )

            def lexical(text: str) -> list[ScoredChunk]:
                return self.lexical_index.search(
                    text,
                    top_k=top_k,
                    principal=principal,
                    index_version=index_version,
                )

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                dense_futures = [pool.submit(dense, v) for v in vectors]
                lexical_futures = [pool.submit(lexical, q) for q in queries]
                dense_hits = [f.result() for f in dense_futures]
                lexical_hits = [f.result() for f in lexical_futures]

            ranked_lists = dense_hits + lexical_hits
            weights = [self.dense_weight] * len(dense_hits) + [self.lexical_weight] * len(
                lexical_hits
            )
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)

            # Defense in depth: re-check ACLs after fusion even though every
            # backend already filtered server-side.
            filtered = [s for s in fused if is_allowed(principal, s.chunk.acl)]
            dropped = len(fused) - len(filtered)
            if dropped:
                attrs["acl_post_filtered"] = dropped

            results = [
                ScoredChunk(chunk=s.chunk, score=s.score, rank=i, retriever="fused")
                for i, s in enumerate(filtered[:top_k], start=1)
            ]
            attrs["dense"] = sum(len(h) for h in dense_hits)
            attrs["lexical"] = sum(len(h) for h in lexical_hits)
            attrs["fused"] = len(results)
            return results
