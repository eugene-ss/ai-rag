from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"Expected mapping in {path}"
        raise TypeError(msg)
    return data


class Settings(BaseSettings):
    """Single source of runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "dev"
    index_version: str = "v1"
    collection_alias: str = "docs_live"
    data_root: Path = Path("data")
    configs_root: Path = Path("configs")
    refusal_score_threshold: float = 0.01
    rrf_k: int = 60
    top_k: int = 10
    rerank_top_k: int = 5
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    llm_max_attempts: int = 3
    llm_timeout_seconds: float = 30.0
    log_level: str = "INFO"
    log_dir: Path | None = None
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    opensearch_url: str = Field(default="http://localhost:9200", validation_alias="OPENSEARCH_URL")

    def chunking_config(self) -> dict[str, Any]:
        return _load_yaml(self.configs_root / "chunking.yaml")

    def retrieval_config(self) -> dict[str, Any]:
        base = _load_yaml(self.configs_root / "retrieval.yaml")
        return {
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "lexical_weight": self.lexical_weight,
            "cache_enabled": self.cache_enabled,
            "semantic_cache_enabled": self.semantic_cache_enabled,
            **base,
        }

    def index_versions(self) -> dict[str, Any]:
        return _load_yaml(self.configs_root / "index_versions.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()
