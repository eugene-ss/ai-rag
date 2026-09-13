from __future__ import annotations

from typing import Any

from rag.exceptions import MissingBackendError
from rag.observability.logging import get_logger
from rag.retrieval.filters import AclFilter
from rag.schemas import AclTags, Chunk, Principal, ScoredChunk

log = get_logger("lexical.opensearch")

_MAPPING: dict[str, Any] = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "ordinal": {"type": "integer"},
            "text": {"type": "text", "analyzer": "standard"},
            "span_start": {"type": "integer"},
            "span_end": {"type": "integer"},
            "tenant": {"type": "keyword"},
            "allow_groups": {"type": "keyword"},
            "classification": {"type": "keyword"},
            "index_version": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": False},
        }
    },
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 1}},
}


class OpenSearchLexicalIndex:
    """OpenSearch BM25 index with server-side ACL filtering.

    Requires the 'opensearch' extra. ACL terms are part of the query filter, so
    unauthorized documents are excluded before scoring.
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:9200",
        index: str = "docs",
        timeout: float = 10.0,
        client: Any | None = None,
    ) -> None:
        try:
            from opensearchpy import OpenSearch
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("OpenSearchLexicalIndex", "opensearch") from exc

        self.index = index
        # Indices are named `<prefix>__<version>`; reads follow the version the
        # caller resolved, not the one this process started with.
        self.prefix = index.rsplit("__", 1)[0] if "__" in index else index
        self._client = client if client is not None else OpenSearch(hosts=[url], timeout=timeout)

    def index_for(self, index_version: str | None = None) -> str:
        """Physical index holding `index_version`."""
        if not index_version:
            return self.index
        return f"{self.prefix}__{index_version}"

    def ensure_index(self) -> None:
        if not self._client.indices.exists(index=self.index):
            self._client.indices.create(index=self.index, body=_MAPPING)

    def upsert(self, chunks: list[Chunk]) -> None:
        from opensearchpy.helpers import bulk

        if not chunks:
            return
        self.ensure_index()
        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": chunk.chunk_id,
                "_source": _to_source(chunk),
            }
            for chunk in chunks
        ]
        bulk(self._client, actions, refresh=True)

    def delete(self, chunk_ids: list[str]) -> None:
        from opensearchpy.helpers import bulk

        if not chunk_ids:
            return
        actions = [{"_op_type": "delete", "_index": self.index, "_id": cid} for cid in chunk_ids]
        bulk(self._client, actions, refresh=True, raise_on_error=False)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": query}}}],
                    "filter": self._acl_filter(
                        AclFilter.for_principal(principal, index_version=index_version)
                    ),
                }
            },
        }
        response = self._client.search(index=self.index_for(index_version), body=body)
        hits = response.get("hits", {}).get("hits", [])
        results: list[ScoredChunk] = []
        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            results.append(
                ScoredChunk(
                    chunk=_from_source(str(hit["_id"]), source),
                    score=float(hit.get("_score") or 0.0),
                    rank=rank,
                    retriever="lexical",
                )
            )
        return results

    def _acl_filter(self, acl: AclFilter) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [{"term": {"tenant": acl.tenant}}]
        if acl.index_version:
            filters.append({"term": {"index_version": acl.index_version}})
        # Visible when tenant-wide (field absent) or sharing at least one group.
        should: list[dict[str, Any]] = [
            {"bool": {"must_not": {"exists": {"field": "allow_groups"}}}}
        ]
        if acl.groups:
            should.append({"terms": {"allow_groups": sorted(acl.groups)}})
        filters.append({"bool": {"should": should, "minimum_should_match": 1}})
        return filters

    def count(self, index_version: str | None = None) -> int:
        return int(self._client.count(index=self.index_for(index_version))["count"])


def _to_source(chunk: Chunk) -> dict[str, Any]:
    source: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "span_start": chunk.span[0],
        "span_end": chunk.span[1],
        "tenant": chunk.acl.tenant,
        "classification": chunk.acl.classification,
        "index_version": chunk.index_version,
        "embedding_model": chunk.embedding_model,
        "metadata": chunk.metadata,
    }
    # Omit the field entirely for tenant-wide chunks so `must_not exists` works.
    if chunk.acl.allow_groups:
        source["allow_groups"] = sorted(chunk.acl.allow_groups)
    return source


def _from_source(doc_id: str, source: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(source.get("chunk_id", doc_id)),
        doc_id=str(source.get("doc_id", "")),
        ordinal=int(source.get("ordinal", 0)),
        text=str(source.get("text", "")),
        span=(int(source.get("span_start", 0)), int(source.get("span_end", 0))),
        acl=AclTags(
            tenant=str(source.get("tenant", "")),
            allow_groups=frozenset(source.get("allow_groups") or ()),
            classification=str(source.get("classification", "internal")),
        ),
        index_version=str(source.get("index_version", "")),
        embedding_model=source.get("embedding_model"),
        metadata=dict(source.get("metadata") or {}),
    )
