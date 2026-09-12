from __future__ import annotations

from dataclasses import dataclass

from rag.pipelines.online import OnlinePipeline, default_online_pipeline
from rag.schemas import Principal
from rag.security.auth import authenticate, principal_from_headers
from rag.settings import Settings, get_settings


@dataclass
class AppState:
    """Shared online-path dependencies. Built once at startup."""

    pipeline: OnlinePipeline
    settings: Settings


_STATE: AppState | None = None


def init_state(
    *,
    pipeline: OnlinePipeline | None = None,
    settings: Settings | None = None,
) -> AppState:
    global _STATE
    settings = settings or get_settings()
    _STATE = AppState(
        pipeline=pipeline or default_online_pipeline(settings=settings),
        settings=settings,
    )
    return _STATE


def get_state() -> AppState:
    if _STATE is None:
        return init_state()
    return _STATE


def resolve_principal(
    authorization: str | None = None,
    x_tenant: str | None = None,
    x_groups: str | None = None,
    x_subject: str | None = None,
) -> Principal:
    if x_subject or x_tenant or x_groups:
        return principal_from_headers(
            subject=x_subject,
            tenant=x_tenant,
            groups=x_groups,
        )
    return authenticate(authorization)
