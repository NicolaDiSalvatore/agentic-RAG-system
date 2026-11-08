"""Langfuse tracing against the real SDK, exporting to memory instead of the network."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.config import settings
from src.llm import langchain_llm as llm
from src.observability import tracing

SECRET = "Where did Phileas Fogg live?"


@pytest.fixture(scope="module")
def memory_client():
    # The SDK keeps one client per public key, so a later client would ignore
    # its own exporter. Every test shares this one.
    spans = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://127.0.0.1:9",
        span_exporter=spans,
        flush_at=1,
        mask=tracing._redact,
    )
    return spans, client


@pytest.fixture
def exporter(memory_client, monkeypatch):
    spans, client = memory_client
    spans.clear()
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    tracing.reset_client(client)
    yield spans, client
    tracing.reset_client(None)


def _exported_text(spans, client) -> str:
    client.flush()
    return json.dumps(
        [{"name": span.name, "attributes": dict(span.attributes)} for span in spans.get_finished_spans()],
        default=str,
    )


def _answer_inside_a_trace(monkeypatch, request_id: str) -> None:
    model = FakeListChatModel(responses=["He lived at Saville Row."])
    target = llm.LLMTarget(provider="groq", model="test-model")
    monkeypatch.setattr(llm, "llm_targets", lambda: [target])
    monkeypatch.setattr(llm, "_get_chat_model", lambda provider, name: model)

    with tracing.query_trace(request_id, "tenant-demo", SECRET) as trace:
        answer = llm.generate_answer(SECRET, None)
        trace.finish({"route": "simple", "fallback": "none"}, {"grounded": 1.0}, output=answer)


def test_query_trace_records_the_llm_call_without_its_content(exporter, monkeypatch):
    spans, client = exporter
    monkeypatch.setattr(settings, "langfuse_capture_query_text", False)

    _answer_inside_a_trace(monkeypatch, "req-1")

    exported = _exported_text(spans, client)
    assert "req-1" in exported
    assert "simple" in exported
    assert "Phileas" not in exported
    assert "Saville" not in exported
    assert "redacted" in exported


def test_each_llm_call_is_one_named_entry(exporter, monkeypatch):
    spans, client = exporter

    _answer_inside_a_trace(monkeypatch, "req-6")

    client.flush()
    assert sorted(span.name for span in spans.get_finished_spans()) == ["generate", "query"]


def test_query_span_shows_text_and_length_while_llm_entries_stay_redacted(exporter, monkeypatch):
    spans, client = exporter

    _answer_inside_a_trace(monkeypatch, "req-8")

    client.flush()
    finished = {span.name: span for span in spans.get_finished_spans()}
    recorded = json.loads(finished["query"].attributes["langfuse.observation.input"])
    assert recorded == {"redacted_text": f"[redacted {len(SECRET)} chars]", "full_text": SECRET}
    assert "Saville" in finished["query"].attributes["langfuse.observation.output"]
    generate = json.dumps(dict(finished["generate"].attributes), default=str)
    assert "Phileas" not in generate
    assert "Saville" not in generate


def test_query_span_records_its_input_output_and_fallback(exporter, monkeypatch):
    spans, client = exporter
    monkeypatch.setattr(settings, "langfuse_capture_query_text", False)

    _answer_inside_a_trace(monkeypatch, "req-7")

    client.flush()
    query = next(span for span in spans.get_finished_spans() if span.name == "query")
    assert query.attributes["langfuse.observation.input"] == f"[redacted {len(SECRET)} chars]"
    assert "redacted" in query.attributes["langfuse.observation.output"]
    assert json.loads(query.attributes["langfuse.observation.metadata.fallback"]) == "none"


def test_tenant_is_hashed_in_traces(exporter):
    spans, client = exporter
    with tracing.query_trace("req-2", "tenant-secret"):
        pass
    assert "tenant-secret" not in _exported_text(spans, client)


def test_errors_propagate_through_the_trace(exporter):
    with pytest.raises(ValueError):
        with tracing.query_trace("req-3", "tenant-demo"):
            raise ValueError("boom")


def test_tracing_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    tracing.reset_client(None)
    assert tracing.callbacks() == []
    with tracing.query_trace("req-4", "tenant-demo") as trace:
        trace.finish({"route": "simple"}, {"grounded": 1.0})


def test_a_broken_langfuse_client_cannot_fail_a_query(monkeypatch):
    class BrokenClient:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(settings, "langfuse_enabled", True)
    tracing.reset_client(BrokenClient())
    try:
        with tracing.query_trace("req-5", "tenant-demo") as trace:
            trace.finish({}, {"grounded": 1.0})
    finally:
        tracing.reset_client(None)
