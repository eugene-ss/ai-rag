"""Authentication, chunk-level ACLs, and PII redaction."""

from rag.security.acl import acl_fingerprint, is_allowed
from rag.security.auth import (
    AnonymousAuthenticator,
    Authenticator,
    AuthError,
    Credentials,
    DenyAllAuthenticator,
    StaticTokenAuthenticator,
    TrustedHeaderAuthenticator,
    build_authenticator,
    parse_groups,
)
from rag.security.pii import redact_pii

__all__ = [
    "AnonymousAuthenticator",
    "AuthError",
    "Authenticator",
    "Credentials",
    "DenyAllAuthenticator",
    "StaticTokenAuthenticator",
    "TrustedHeaderAuthenticator",
    "acl_fingerprint",
    "build_authenticator",
    "is_allowed",
    "parse_groups",
    "redact_pii",
]
