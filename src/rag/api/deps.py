from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from rag.api.errors import UnauthorizedError
from rag.pipelines.online import OnlinePipeline, default_online_pipeline
from rag.schemas import Principal
from rag.security.auth import Authenticator, AuthError, Credentials, build_authenticator
from rag.settings import Settings, get_settings


@dataclass
class AppState:
    """Online-path dependencies, built once during startup."""

    pipeline: OnlinePipeline
    settings: Settings
    authenticator: Authenticator


def build_state(
    *,
    pipeline: OnlinePipeline | None = None,
    settings: Settings | None = None,
    authenticator: Authenticator | None = None,
) -> AppState:
    settings = settings or get_settings()
    return AppState(
        pipeline=pipeline or default_online_pipeline(settings),
        settings=settings,
        authenticator=authenticator or build_authenticator(settings),
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
