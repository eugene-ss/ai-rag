from __future__ import annotations

from typing import Any

from rag.exceptions import MissingBackendError
from rag.observability.logging import get_logger
from rag.retrieval.filters import AclFilter
from rag.schemas import AclTags, Chunk, Principal, ScoredChunk

log = get_logger("vectordb.qdrant")

_PAYLOAD_FIELDS = (
    "doc_id",
    "ordinal",
    "text",
    "span_start",
    "span_end",
    "tenant",
    "allow_groups",
    "classification",
    "index_version",
    "embedding_model",
    "metadata",
)


class QdrantVectorStore:
    """Qdrant-backed dense index with server-side ACL filtering.

    Requires the 'qdrant' extra. Filters are pushed into the Qdrant query so
    unauthorized vectors are never scored, let alone returned.
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection: str = "docs",
        dimensions: int = 1536,
        alias_collection: str | None = None,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise MissingBackendError("QdrantVectorStore", "qdrant") from exc

        self.collection = collection
        # Physical collections are named `<prefix>__<version>`. Keep the prefix so
        # reads can be routed to whichever version the alias currently publishes,
        # rather than to whichever version this process happened to start with.
        self.prefix = collection.rsplit("__", 1)[0] if "__" in collection else collection
        self.dimensions = dimensions
        self._alias_collection = alias_collection or f"{collection}__aliases"
        self._client = client if client is not None else QdrantClient(url=url, api_key=api_key)
        self._aliases: dict[str, str] = {}

    def collection_for(self, index_version: str | None = None) -> str:
        """Physical collection holding `index_version`.

        Writes target the version being built (`self.collection`); reads target
        the version the caller resolved from the alias. Without this, promoting
        `v2` leaves the API querying the `v1` collection while filtering for
        `index_version=v2`, which matches nothing and looks like an empty corpus.
        """
        if not index_version:
            return self.collection
        return f"{self.prefix}__{index_version}"

    # --- schema ------------------------------------------------------------

    def ensure_collection(self, collection: str | None = None) -> None:
        """Create the collection and ACL payload indexes if absent."""
        from qdrant_client import models

        name = collection or self.collection
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=self.dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        # Payload indexes keep ACL filtering cheap at scale.
        for field, schema in (
            ("tenant", models.PayloadSchemaType.KEYWORD),
            ("allow_groups", models.PayloadSchemaType.KEYWORD),
            ("index_version", models.PayloadSchemaType.KEYWORD),
        ):
            self._client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )

    # --- writes ------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client import models

        if len(chunks) != len(vectors):
            msg = "chunks and vectors length mismatch"
            raise ValueError(msg)
        if not chunks:
            return
        self.ensure_collection()
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete(self, chunk_ids: list[str]) -> None:
        from qdrant_client import models

        if not chunk_ids:
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[_point_id(c) for c in chunk_ids]),
            wait=True,
        )

    # --- reads -------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        principal: Principal,
        index_version: str | None = None,
    ) -> list[ScoredChunk]:
        response = self._client.query_points(
            collection_name=self.collection_for(index_version),
            query=vector,
            limit=top_k,
            query_filter=self._acl_filter(
                AclFilter.for_principal(principal, index_version=index_version)
            ),
            with_payload=True,
        )
        results: list[ScoredChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            results.append(
                ScoredChunk(
                    chunk=_from_payload(str(point.id), payload),
                    score=float(point.score),
                    rank=rank,
                    retriever="dense",
                )
            )
        return results

    def _acl_filter(self, acl: AclFilter) -> Any:
        """Tenant match plus group overlap, evaluated inside Qdrant."""
        from qdrant_client import models

        must: list[Any] = [
            models.FieldCondition(
                key="tenant",
                match=models.MatchValue(value=acl.tenant),
            )
        ]
        if acl.index_version:
            must.append(
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchValue(value=acl.index_version),
                )
            )
        # A chunk is visible when it is tenant-wide (no groups) or shares a group.
        # The alternatives belong *inside* MinShould: `min_should` is its own
        # clause taking `conditions` + `min_count`, not an int paired with
        # `should`. Passing an int raises ValidationError on every search.
        group_clauses: list[Any] = [
            models.IsEmptyCondition(is_empty=models.PayloadField(key="allow_groups"))
        ]
        if acl.groups:
            group_clauses.append(
                models.FieldCondition(
                    key="allow_groups",
                    match=models.MatchAny(any=sorted(acl.groups)),
                )
            )
        return models.Filter(
            must=must,
            min_should=models.MinShould(conditions=group_clauses, min_count=1),
        )

    # --- aliases -----------------------------------------------------------

    def resolve_alias(self, alias: str) -> str | None:
        """Version currently published under `alias`, read from Qdrant."""
        try:
            records = self._client.get_aliases().aliases
        except Exception as exc:  # pragma: no cover - transport dependent
            log.warning("could not read qdrant aliases: %s", exc)
            return self._aliases.get(alias)
        for record in records:
            if record.alias_name == alias:
                return str(record.collection_name).rsplit("__", 1)[-1]
        return self._aliases.get(alias)

    def set_alias(self, alias: str, index_version: str) -> None:
        """Publish `index_version` under `alias` in one atomic operation batch.

        Delete-then-create in a single call, because creating an alias that
        already exists is not a move — rollback to a previous version would fail
        against a live cluster.
        """
        from qdrant_client import models

        target = self.collection_for(index_version)
        self._aliases[alias] = index_version
        self._client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=target, alias_name=alias)
                ),
            ]
        )
        log.info("qdrant alias %s -> %s", alias, target)

    def count(self, index_version: str | None = None) -> int:
        return int(
            self._client.count(collection_name=self.collection_for(index_version), exact=True).count
        )


def _point_id(chunk_id: str) -> str:
    import uuid

    # Qdrant ids must be UUIDs or ints; chunk_id is a stable hex digest.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _to_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "span_start": chunk.span[0],
        "span_end": chunk.span[1],
        "tenant": chunk.acl.tenant,
        "allow_groups": sorted(chunk.acl.allow_groups),
        "classification": chunk.acl.classification,
        "index_version": chunk.index_version,
        "embedding_model": chunk.embedding_model,
        "metadata": chunk.metadata,
    }


def _from_payload(point_id: str, payload: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(payload.get("chunk_id", point_id)),
        doc_id=str(payload.get("doc_id", "")),
        ordinal=int(payload.get("ordinal", 0)),
        text=str(payload.get("text", "")),
        span=(int(payload.get("span_start", 0)), int(payload.get("span_end", 0))),
        acl=AclTags(
            tenant=str(payload.get("tenant", "")),
            allow_groups=frozenset(payload.get("allow_groups") or ()),
            classification=str(payload.get("classification", "internal")),
        ),
        index_version=str(payload.get("index_version", "")),
        embedding_model=payload.get("embedding_model"),
        metadata=dict(payload.get("metadata") or {}),
    )
