from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from rag.observability.logging import get_logger
from rag.schemas import Principal
from rag.settings import Settings

log = get_logger("security.auth")


class AuthError(Exception):
    """Caller could not be authenticated. Surfaces as HTTP 401."""

    def __init__(self, reason: str = "unauthenticated") -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Credentials:
    """Raw, untrusted material extracted from the transport layer."""

    authorization: str | None = None
    tenant: str | None = None
    groups: str | None = None
    subject: str | None = None


@runtime_checkable
class Authenticator(Protocol):
    """Turns transport credentials into a trusted Principal."""

    name: str

    def authenticate(self, credentials: Credentials) -> Principal: ...


def parse_groups(raw: str | None, *, default: frozenset[str] = frozenset()) -> frozenset[str]:
    if not raw:
        return default
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


class TrustedHeaderAuthenticator:
    """Takes identity straight from request headers.

    INSECURE BY DESIGN: any caller can name their own tenant and groups, so this
    is only constructible when `auth_trust_headers` is enabled, which Settings
    forbids outside dev.
    """

    name = "trusted_header"

    def __init__(self, *, default_tenant: str = "default") -> None:
        self.default_tenant = default_tenant

    def authenticate(self, credentials: Credentials) -> Principal:
        subject = credentials.subject or _bearer_token(credentials.authorization) or "anonymous"
        return Principal(
            subject=subject,
            tenant=credentials.tenant or self.default_tenant,
            groups=parse_groups(credentials.groups, default=frozenset({"public"})),
        )


class StaticTokenAuthenticator:
    """Maps opaque bearer tokens to principals from a YAML file.

    Tokens are compared in constant time and never logged. Intended as the
    default production seam until a real IdP (OIDC/JWT) is wired in; swap this
    class out rather than loosening it.
    """

    name = "static_token"

    def __init__(self, principals: dict[str, Principal]) -> None:
        self._principals = principals

    @classmethod
    def from_file(cls, path: Path | str) -> StaticTokenAuthenticator:
        raw = Path(path).read_text()
        data: Any = yaml.safe_load(raw) or {}
        entries = data.get("tokens", []) if isinstance(data, dict) else []
        principals: dict[str, Principal] = {}
        for entry in entries:
            token = str(entry["token"])
            principals[token] = Principal(
                subject=str(entry["subject"]),
                tenant=str(entry["tenant"]),
                groups=frozenset(str(g) for g in entry.get("groups", [])),
            )
        if not principals:
            log.warning("no tokens loaded from %s; all requests will be rejected", path)
        return cls(principals)

    def authenticate(self, credentials: Credentials) -> Principal:
        token = _bearer_token(credentials.authorization)
        if not token:
            raise AuthError("missing_bearer_token")
        import hmac

        for known, principal in self._principals.items():
            if hmac.compare_digest(token, known):
                return principal
        raise AuthError("invalid_token")


class DenyAllAuthenticator:
    """Rejects everything. Used when auth is required but unconfigured."""

    name = "deny_all"

    def authenticate(self, credentials: Credentials) -> Principal:
        raise AuthError("no_authenticator_configured")


class AnonymousAuthenticator:
    """Single anonymous principal. Only for explicitly unauthenticated demos."""

    name = "anonymous"

    def __init__(self, *, tenant: str = "default") -> None:
        self.tenant = tenant

    def authenticate(self, credentials: Credentials) -> Principal:
        return Principal(
            subject="anonymous",
            tenant=self.tenant,
            groups=frozenset({"public"}),
        )


def build_authenticator(settings: Settings) -> Authenticator:
    """Select an authenticator from configuration, failing closed."""
    if settings.auth_tokens_file is not None:
        return StaticTokenAuthenticator.from_file(settings.auth_tokens_file)
    if settings.auth_trust_headers:
        log.warning(
            "auth_trust_headers is enabled: ACLs are caller-supplied and unverified (env=%s)",
            settings.env,
        )
        return TrustedHeaderAuthenticator(default_tenant=settings.default_tenant)
    if not settings.auth_required:
        log.warning("auth_required is disabled: serving all requests as anonymous")
        return AnonymousAuthenticator(tenant=settings.default_tenant)
    return DenyAllAuthenticator()
