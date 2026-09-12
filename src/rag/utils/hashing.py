from __future__ import annotations

import hashlib


def sha256_hex(*parts: str | bytes) -> str:
    """Stable hash over the given parts, used for ids and cache keys."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else part.encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def short_id(*parts: str | bytes, length: int = 24) -> str:
    return sha256_hex(*parts)[:length]
