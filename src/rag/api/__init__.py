"""FastAPI surface for the online query path only."""

from rag.api.app import app, create_app
from rag.api.deps import AppState, build_state

__all__ = ["AppState", "app", "build_state", "create_app"]
