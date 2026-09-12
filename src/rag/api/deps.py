from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Header, HTTPException, Request, status

from rag.api.errors import UnauthorizedError
from rag.backends import build_agent_runtime
from rag.pipelines.online import OnlinePipeline, default_online_pipeline
from rag.schemas import Principal
from rag.security.auth import Authenticator, AuthError, Credentials, build_authenticator
from rag.settings import Settings, get_settings

if TYPE_CHECKING:
    from rag.agent.runtime import AgentRuntime


@dataclass
class AppState:
    """Online-path dependencies, built once during startup."""

    pipeline: OnlinePipeline
    settings: Settings
    authenticator: Authenticator
    agent: AgentRuntime | None = None
    _tenant_semaphores: dict[str, asyncio.Semaphore] = field(default_factory=dict)

    def agent_semaphore_for(self, tenant: str) -> asyncio.Semaphore:
        """Per-tenant concurrency cap — budget alone is not enough against stampedes."""
        if tenant not in self._tenant_semaphores:
            limit = max(1, self.settings.agent_max_concurrent_per_tenant)
            self._tenant_semaphores[tenant] = asyncio.Semaphore(limit)
        return self._tenant_semaphores[tenant]


def build_state(
    *,
    pipeline: OnlinePipeline | None = None,
    settings: Settings | None = None,
    authenticator: Authenticator | None = None,
    agent: AgentRuntime | None = None,
) -> AppState:
    settings = settings or get_settings()
    pipeline = pipeline or default_online_pipeline(settings)
    if agent is None and settings.agent_enabled:
        agent = build_agent_runtime(settings, pipeline)
    return AppState(
        pipeline=pipeline,
        settings=settings,
        authenticator=authenticator or build_authenticator(settings),
        agent=agent,
    )


def get_state(request: Request) -> AppState:
    """Read state off the app rather than a module global.

    Keeps tests isolated and makes multiple app instances in one process safe.
    """
    state: AppState | None = getattr(request.app.state, "rag", None)
    if state is None:  # pragma: no cover - startup always sets this
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service not initialised",
        )
    return state


def get_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_tenant: str | None = Header(default=None),
    x_groups: str | None = Header(default=None),
    x_subject: str | None = Header(default=None),
) -> Principal:
    """Authenticate the caller, or fail the request.

    ACL-bearing headers are only honoured by TrustedHeaderAuthenticator, which
    Settings refuses to enable outside dev.
    """
    state = get_state(request)
    credentials = Credentials(
        authorization=authorization,
        tenant=x_tenant,
        groups=x_groups,
        subject=x_subject,
    )
    try:
        return state.authenticator.authenticate(credentials)
    except AuthError as exc:
        raise UnauthorizedError(exc.reason) from exc
