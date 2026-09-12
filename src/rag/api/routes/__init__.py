"""Online query path only — no ingestion or indexing."""

from rag.api.routes import health, query

__all__ = ["health", "query"]
