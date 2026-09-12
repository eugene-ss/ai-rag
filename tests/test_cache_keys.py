from __future__ import annotations

from rag.cache.keys import cache_key
from rag.schemas import Principal


def test_cache_key_differs_across_tenants() -> None:
    a = Principal(subject="u1", tenant="acme", groups=frozenset({"eng"}))
    b = Principal(subject="u1", tenant="beta", groups=frozenset({"eng"}))
    ka = cache_key(
        query="What is hybrid retrieval?",
        principal=a,
        index_version="v1",
        prompt_version="v1",
    )
    kb = cache_key(
        query="What is hybrid retrieval?",
        principal=b,
        index_version="v1",
        prompt_version="v1",
    )
    assert ka != kb


def test_cache_key_differs_across_groups() -> None:
    a = Principal(subject="u1", tenant="acme", groups=frozenset({"eng"}))
    b = Principal(subject="u1", tenant="acme", groups=frozenset({"hr"}))
    ka = cache_key(query="q", principal=a, index_version="v1", prompt_version="v1")
    kb = cache_key(query="q", principal=b, index_version="v1", prompt_version="v1")
    assert ka != kb


def test_cache_key_stable_for_same_acl() -> None:
    a = Principal(subject="u1", tenant="acme", groups=frozenset({"eng", "ops"}))
    b = Principal(subject="u1", tenant="acme", groups=frozenset({"ops", "eng"}))
    ka = cache_key(query="  Hello   World ", principal=a, index_version="v1", prompt_version="v1")
    kb = cache_key(query="hello world", principal=b, index_version="v1", prompt_version="v1")
    assert ka == kb
