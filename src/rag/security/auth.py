from __future__ import annotations

from rag.schemas import Principal


def principal_from_headers(
    *,
    subject: str | None = None,
    tenant: str | None = None,
    groups: str | None = None,
) -> Principal:
    """Build a Principal from HTTP headers / request context.

    Production systems should replace this with real JWT / IAM validation.
    """
    group_set = frozenset(g.strip() for g in (groups or "public").split(",") if g.strip())
    return Principal(
        subject=subject or "anonymous",
        tenant=tenant or "default",
        groups=group_set,
    )


def authenticate(authorization: str | None = None) -> Principal:
    """Minimal auth seam. Treats 'Bearer <subject>' as identity; else anonymous."""
    if not authorization:
        return principal_from_headers()
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return principal_from_headers(subject=parts[1])
    return principal_from_headers(subject=authorization)
