"""API contract tests for auth, readiness, limits, and citation output.

The graph is replaced with a stub so these tests never load models or call a
provider; they assert the HTTP surface only.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.api.main as api
from src.resilience import DeadlineExceeded, LLMUnavailable, current_context


RESULT = {
    "answer": "Fogg lived at Number 7.",
    "route": "simple",
    "grounded": True,
    "citations": [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "source_uri": "Around_the_World_-_Jules_Verne",
        }
    ],
    "answer_status": "ok",
    "fallback": None,
}


class StubGraph:
    """Stand-in for the compiled LangGraph app."""

    def __init__(self, result=None, delay=0.0, error=None, gate=None):
        self.result = dict(result or RESULT)
        self.delay = delay
        self.error = error
        self.gate = gate
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        if self.gate is not None and not self.gate.wait(5):
            raise RuntimeError("test gate was never released")
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return dict(self.result)


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def ready(monkeypatch):
    """Pretend a healthy index is present and report a fixed version."""
    monkeypatch.setattr(api, "index_is_ready", lambda: True)
    monkeypatch.setattr(api, "active_index_version", lambda: "v-test")


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_readiness_reports_missing_index(client, monkeypatch):
    monkeypatch.setattr(api, "index_is_ready", lambda: False)
    response = client.get("/health/ready")
    assert response.status_code == 503


def test_readiness_reports_ready_index(client, ready):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "index_version": "v-test"}


def test_query_returns_canonical_contract(client, ready, monkeypatch):
    stub = StubGraph()
    monkeypatch.setattr(api, "graph", stub)

    response = client.post("/query", json={"question": "Where did Fogg live?"})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "answer": RESULT["answer"],
        "route": "simple",
        "grounded": True,
        "citations": RESULT["citations"],
        "index_version": "v-test",
        "answer_status": "ok",
        "fallback": None,
    }
    # The old raw-chunk field must not come back.
    assert "sources" not in body
    # Claims from the API key are passed into the graph.
    assert stub.calls[0]["tenant_id"] == "tenant-demo"
    assert stub.calls[0]["roles"] == ["public"]


def test_request_id_is_echoed_or_generated(client):
    echoed = client.get("/health", headers={"x-request-id": "req-123"})
    assert echoed.headers["x-request-id"] == "req-123"

    generated = client.get("/health")
    assert generated.headers["x-request-id"]


@pytest.mark.parametrize("question", ["", "   ", "x" * (api.settings.max_question_chars + 1)])
def test_invalid_question_is_rejected(client, question):
    assert client.post("/query", json={"question": question}).status_code == 422


def test_auth_rejects_missing_and_wrong_keys(client, ready, monkeypatch):
    monkeypatch.setattr(api.settings, "require_auth", True)
    monkeypatch.setattr(api.settings, "api_key", "secret")
    monkeypatch.setattr(api, "graph", StubGraph())

    assert client.post("/query", json={"question": "q"}).status_code == 401
    wrong = client.post(
        "/query", json={"question": "q"}, headers={"Authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401
    assert wrong.headers["www-authenticate"] == "Bearer"

    ok = client.post(
        "/query", json={"question": "q"}, headers={"Authorization": "Bearer secret"}
    )
    assert ok.status_code == 200


def test_auth_runs_before_the_query_slot_is_taken(client, monkeypatch):
    monkeypatch.setattr(api.settings, "require_auth", True)
    monkeypatch.setattr(api.settings, "api_key", "secret")
    monkeypatch.setattr(api, "query_slots", threading.Semaphore(0))

    response = client.post("/query", json={"question": "q"})

    # An unauthenticated request must be rejected as 401, not 429.
    assert response.status_code == 401


def test_busy_service_returns_429(client, ready, monkeypatch):
    monkeypatch.setattr(api, "graph", StubGraph())
    monkeypatch.setattr(api, "query_slots", threading.Semaphore(0))

    assert client.post("/query", json={"question": "q"}).status_code == 429


def test_timeout_returns_504_and_holds_slot_until_work_ends(client, ready, monkeypatch):
    gate = threading.Event()
    stub = StubGraph(gate=gate)
    monkeypatch.setattr(api, "graph", stub)
    monkeypatch.setattr(api.settings, "request_timeout_seconds", 0.05)
    monkeypatch.setattr(api, "DEADLINE_GRACE_SECONDS", 0.0)

    first = client.post("/query", json={"question": "slow"})
    assert first.status_code == 504

    # The graph thread is still blocked, so the only slot is still in use.
    assert client.post("/query", json={"question": "second"}).status_code == 429

    gate.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if client.post("/query", json={"question": "third"}).status_code == 200:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the query slot was never released after the graph finished")


def test_missing_index_is_a_503(client, ready, monkeypatch):
    monkeypatch.setattr(
        api,
        "graph",
        StubGraph(error=RuntimeError("Qdrant collection is missing")),
    )

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 503
    assert "missing" in response.json()["detail"]


def test_graph_deadline_is_a_504(client, ready, monkeypatch):
    monkeypatch.setattr(api, "graph", StubGraph(error=DeadlineExceeded("late")))
    assert client.post("/query", json={"question": "q"}).status_code == 504


def test_llm_failure_without_evidence_is_a_retryable_503(client, ready, monkeypatch):
    error = LLMUnavailable("all_providers_failed")
    monkeypatch.setattr(api, "graph", StubGraph(error=error))

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 503
    assert response.headers["retry-after"]
    # Provider details stay in the logs.
    assert response.json()["detail"] == "The language model is temporarily unavailable."


def test_request_gets_a_deadline_and_call_budget(client, ready, monkeypatch):
    seen = {}

    class CapturingGraph(StubGraph):
        def invoke(self, state):
            context = current_context()
            seen["deadline"] = context.remaining()
            seen["budget"] = context.max_llm_calls
            return super().invoke(state)

    monkeypatch.setattr(api, "graph", CapturingGraph())
    assert client.post("/query", json={"question": "q"}).status_code == 200
    assert 0 < seen["deadline"] <= api.settings.request_timeout_seconds
    assert seen["budget"] == api.settings.llm_max_calls_per_query


def test_busy_response_says_when_to_retry(client, ready, monkeypatch):
    monkeypatch.setattr(api, "query_slots", threading.Semaphore(0))
    assert client.post("/query", json={"question": "q"}).headers["retry-after"]


def test_metrics_require_the_key_when_auth_is_on(client, monkeypatch):
    monkeypatch.setattr(api.settings, "require_auth", True)
    monkeypatch.setattr(api.settings, "api_key", "secret")
    assert client.get("/metrics").status_code == 401
    allowed = client.get("/metrics", headers={"Authorization": "Bearer secret"})
    assert allowed.status_code == 200


def test_metrics_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(api.settings, "metrics_enabled", False)
    assert client.get("/metrics").status_code == 404


def test_warming_up_service_rejects_queries_and_is_not_ready(client, monkeypatch):
    monkeypatch.setattr(api, "index_is_ready", lambda: True)
    monkeypatch.setattr(api, "_warmup_state", "running")
    assert client.get("/health/ready").status_code == 503
    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 503
    assert response.headers["retry-after"]


def test_failed_warmup_fails_readiness(client, monkeypatch):
    monkeypatch.setattr(api, "index_is_ready", lambda: True)
    monkeypatch.setattr(api, "_warmup_state", "failed")
    assert client.get("/health/ready").status_code == 503


def test_warmup_builds_the_retriever_ahead_of_requests(monkeypatch):
    retriever = MagicMock()
    monkeypatch.setattr("src.graph.nodes._get_retriever", lambda tenant, roles: retriever)
    monkeypatch.setattr(api, "_warmup_state", "running")
    api._warm_up()
    retriever.retrieve.assert_called_once()
    assert api._warmup_state == "ready"


def test_startup_fails_when_auth_has_no_key(monkeypatch):
    monkeypatch.setattr(api.settings, "require_auth", True)
    monkeypatch.setattr(api.settings, "api_key", "")

    async def run_lifespan():
        async with api.lifespan(api.app):
            pass

    with pytest.raises(RuntimeError, match="API_KEY"):
        asyncio.run(run_lifespan())
