from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorBackend = Literal["memory", "qdrant"]
LexicalBackend = Literal["memory", "opensearch"]
EmbeddingBackend = Literal["hash", "openai"]
LLMBackend = Literal["echo", "openai"]
CacheBackend = Literal["memory", "redis"]
RerankerName = Literal["identity", "cross_encoder"]

PRODUCTION_ENVS = frozenset({"prod", "production", "staging", "stage"})


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"Expected mapping in {path}"
        raise TypeError(msg)
    return data


class RetrievalConfig(BaseModel):
    """Resolved retrieval knobs for one query.

    A typed view over configs/retrieval.yaml merged with environment overrides,
    so the pipeline never indexes into an untyped dict at runtime.
    """

    top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rewrite_enabled: bool = True
    multi_query_enabled: bool = False
    multi_query_count: int = 3
    hyde_enabled: bool = False
    cache_enabled: bool = True
    semantic_cache_enabled: bool = True

    def cache_params(self) -> dict[str, Any]:
        """Fields that must change the cache key when they change."""
        return self.model_dump(
            include={
                "top_k",
                "rerank_top_k",
                "rrf_k",
                "dense_weight",
                "lexical_weight",
                "rewrite_enabled",
                "multi_query_enabled",
                "multi_query_count",
                "hyde_enabled",
            }
        )


class Settings(BaseSettings):
    """Single source of runtime configuration.

    Every value is overridable by a `RAG_`-prefixed environment variable, except
    third-party credentials which keep their conventional names (OPENAI_API_KEY,
    QDRANT_URL, OPENSEARCH_URL, REDIS_URL).
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- environment -------------------------------------------------------
    env: str = "dev"
    service_name: str = "rag-api"

    # --- index identity ----------------------------------------------------
    index_version: str = "v1"
    collection_alias: str = "docs_live"
    collection_prefix: str = "docs"
    data_root: Path = Path("data")
    configs_root: Path = Path("configs")
    # Override to keep the registry off the config mount (tests, read-only FS).
    index_registry_file: Path | None = None

    # --- backend selection -------------------------------------------------
    vector_backend: VectorBackend = "memory"
    lexical_backend: LexicalBackend = "memory"
    embedding_backend: EmbeddingBackend = "hash"
    llm_backend: LLMBackend = "echo"
    cache_backend: CacheBackend = "memory"
    reranker: RerankerName = "identity"

    # --- retrieval ---------------------------------------------------------
    top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    rewrite_enabled: bool = True
    multi_query_enabled: bool = False
    multi_query_count: int = 3
    hyde_enabled: bool = False
    refusal_score_threshold: float = 0.01

    # --- caching -----------------------------------------------------------
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95

    # --- models ------------------------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    hash_embedding_dimensions: int = 64
    llm_model: str = "gpt-4o-mini"
    llm_fallback_models: list[str] = Field(default_factory=list)
    llm_max_attempts: int = 3
    llm_timeout_seconds: float = 30.0
    llm_max_output_tokens: int = 1024
    llm_temperature: float = 0.0

    # --- security ----------------------------------------------------------
    auth_required: bool = True
    # Trusting X-Tenant/X-Groups lets any caller pick their own ACL. Local only.
    auth_trust_headers: bool = False
    auth_tokens_file: Path | None = None
    default_tenant: str = "default"
    pii_redaction_enabled: bool = True

    # --- api ---------------------------------------------------------------
    cors_allow_origins: list[str] = Field(default_factory=list)
    request_timeout_seconds: float = 30.0
    metrics_enabled: bool = True

    # --- observability -----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False
    log_dir: Path | None = None

    # --- credentials / endpoints -------------------------------------------
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    opensearch_url: str = Field(default="http://localhost:9200", validation_alias="OPENSEARCH_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")

    @field_validator("llm_fallback_models", "cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept comma-separated env values as well as JSON lists."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def _reject_unsafe_production_config(self) -> Settings:
        """Fail fast rather than silently running an unsafe production config."""
        if not self.is_production:
            return self
        problems: list[str] = []
        if self.auth_trust_headers:
            problems.append(
                "RAG_AUTH_TRUST_HEADERS must be false outside dev: trusting ACL headers "
                "lets any caller choose their own tenant and groups"
            )
        if not self.auth_required:
            problems.append("RAG_AUTH_REQUIRED must be true outside dev")
        if self.embedding_backend == "hash":
            problems.append(
                "RAG_EMBEDDING_BACKEND=hash is a deterministic test stub, not a real model"
            )
        if self.llm_backend == "echo":
            problems.append("RAG_LLM_BACKEND=echo is a test stub, not a real model")
        if problems:
            joined = "\n  - ".join(problems)
            msg = f"Unsafe configuration for env={self.env!r}:\n  - {joined}"
            raise ValueError(msg)
        return self

    @property
    def is_production(self) -> bool:
        return self.env.lower() in PRODUCTION_ENVS

    def collection_for(self, index_version: str | None = None) -> str:
        """Physical collection name for a version, e.g. docs__v2."""
        return f"{self.collection_prefix}__{index_version or self.index_version}"

    def chunking_config(self) -> dict[str, Any]:
        return _load_yaml(self.configs_root / "chunking.yaml")

    def retrieval_config(self) -> RetrievalConfig:
        """Retrieval knobs. YAML provides defaults; env vars always win."""
        base = _load_yaml(self.configs_root / "retrieval.yaml")
        known = set(RetrievalConfig.model_fields)
        overrides = self.model_dump(include=known & set(type(self).model_fields))
        return RetrievalConfig.model_validate(
            {**{k: v for k, v in base.items() if k in known}, **overrides}
        )

    def index_registry_path(self) -> Path:
        return self.index_registry_file or (self.configs_root / "index_versions.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached Settings; used by tests that patch the environment."""
    get_settings.cache_clear()
