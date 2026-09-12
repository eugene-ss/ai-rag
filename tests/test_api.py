from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.api.deps import AppState, build_state
from rag.backends import BackendContext
from rag.index_registry import IndexRegistry
from rag.pipelines import offline as offline_mod
from rag.pipelines import online as online_mod
from rag.schemas import AclTags, Principal
from rag.security.auth import (
    AuthError,
    Credentials,
    DenyAllAuthenticator,
    StaticTokenAuthenticator,
    TrustedHeaderAuthenticator,
)
from tests.conftest import CORPUS, GROUP, TENANT


@pytest.fixture
def state(context: BackendContext, acl: AclTags) -> AppState:
    """API state over a pre-built index: the API serves, it never indexes."""
    pipeline = offline_mod.from_context(context)
    pipeline.registry = IndexRegistry(context.settings.index_registry_path())
    pipeline.run(CORPUS, index_version="v1", acl=acl)

    return build_state(
        pipeline=online_mod.from_context(context),
        settings=context.settings,
        authenticator=TrustedHeaderAuthenticator(default_tenant=TENANT),
    )


@pytest.fixture
def client(state: AppState) -> Iterator[TestClient]:
    with TestClient(create_app(state=state)) as c:
        yield c


AUTH = {"x-tenant": TENANT, "x-groups": GROUP, "x-subject": "alice"}


def test_health_is_liveness_only(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_live_index(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["index_version"] == "v1"
    assert body["checks"] == {"index_alias": True, "index_populated": True}
    assert body["chunk_count"] > 0


def test_ready_fails_on_empty_index(context: BackendContext) -> None:
    """A replica with no live index must not receive traffic."""
    state = build_state(
        pipeline=online_mod.from_context(context),
        settings=context.settings,
        authenticator=TrustedHeaderAuthenticator(default_tenant=TENANT),
    )
    with TestClient(create_app(state=state)) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["index_populated"] is False
    # Liveness must still pass: the process is healthy, just not ready.
    with TestClient(create_app(state=state)) as client:
        assert client.get("/health").status_code == 200


def test_metrics_endpoint_exposes_counters(client: TestClient) -> None:
    client.post("/query", json={"query": "What is hybrid retrieval?"}, headers=AUTH)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "counters" in resp.json()


def test_query_returns_grounded_answer(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "What is hybrid retrieval?"}, headers=AUTH)
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert answer["refused"] is False
    assert answer["citations"]
    assert answer["index_version"] == "v1"


def test_response_echoes_request_id(client: TestClient) -> None:
    resp = client.post(
        "/query",
        json={"query": "What is hybrid retrieval?"},
        headers={**AUTH, "X-Request-Id": "trace-abc-123"},
    )
    assert resp.headers["X-Request-Id"] == "trace-abc-123"
    assert resp.json()["answer"]["trace_id"] == "trace-abc-123"


def test_cross_tenant_request_is_refused_not_leaked(client: TestClient) -> None:
    """Wrong tenant retrieves nothing, so the only honest answer is refusal."""
    resp = client.post(
        "/query",
        json={"query": "What is hybrid retrieval?"},
        headers={"x-tenant": "other-corp", "x-groups": GROUP, "x-subject": "bob"},
    )
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert answer["refused"] is True
    assert answer["citations"] == []
    assert "hybrid" not in answer["text"].lower()


def test_empty_query_is_rejected_by_validation(client: TestClient) -> None:
    assert client.post("/query", json={"query": ""}, headers=AUTH).status_code == 422


def test_oversized_query_is_rejected(client: TestClient) -> None:
    resp = client.post("/query", json={"query": "x" * 5000}, headers=AUTH)
    assert resp.status_code == 422


def test_unauthenticated_request_is_rejected_when_auth_required(
    context: BackendContext, acl: AclTags
) -> None:
    """Default posture is fail-closed: no credentials means 401, not anonymous."""
    state = build_state(
        pipeline=online_mod.from_context(context),
        settings=context.settings,
        authenticator=DenyAllAuthenticator(),
    )
    with TestClient(create_app(state=state)) as client:
        resp = client.post("/query", json={"query": "anything"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_static_token_authenticator_maps_tokens_to_principals(tmp_path: Path) -> None:
    tokens = tmp_path / "auth.yaml"
    tokens.write_text(
        "tokens:\n"
        "  - token: secret-acme\n"
        "    subject: svc-acme\n"
        "    tenant: acme\n"
        "    groups: [engineering]\n"
    )
    auth = StaticTokenAuthenticator.from_file(tokens)

    who = auth.authenticate(Credentials(authorization="Bearer secret-acme"))
    assert who == Principal(subject="svc-acme", tenant="acme", groups=frozenset({"engineering"}))

    with pytest.raises(AuthError):
        auth.authenticate(Credentials(authorization="Bearer wrong"))
    with pytest.raises(AuthError):
        auth.authenticate(Credentials())


def test_static_token_ignores_acl_headers(tmp_path: Path) -> None:
    """The whole point: a caller cannot escalate by sending X-Tenant."""
    tokens = tmp_path / "auth.yaml"
    tokens.write_text(
        "tokens:\n"
        "  - token: secret-beta\n"
        "    subject: svc-beta\n"
        "    tenant: beta\n"
        "    groups: [public]\n"
    )
    auth = StaticTokenAuthenticator.from_file(tokens)
    who = auth.authenticate(
        Credentials(
            authorization="Bearer secret-beta",
            tenant="acme",
            groups="hr,admin",
            subject="attacker",
        )
    )
    assert who.tenant == "beta"
    assert who.groups == frozenset({"public"})
    assert who.subject == "svc-beta"
