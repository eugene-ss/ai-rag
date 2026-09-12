"""Document ingestion connectors."""

from rag.ingestion.base import SourceConnector
from rag.ingestion.local_fs import LocalFSConnector
from rag.ingestion.registry import default_connector, get_connector, register

__all__ = [
    "LocalFSConnector",
    "SourceConnector",
    "default_connector",
    "get_connector",
    "register",
]
