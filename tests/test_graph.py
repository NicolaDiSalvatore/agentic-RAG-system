"""Tests for the node functions in src/graph/nodes.py and src/graph/build_graph.py."""

from unittest.mock import MagicMock, patch
import pytest
from src.config import settings
from src.graph.build_graph import build_graph
from src.graph.nodes import (
    MAX_RETRIES,
    decompose_node,
    escalate_node,
    generate_node,
    grade_node,
    retrieve_node,
    route_node,
)
from src.resilience import LLMUnavailable


class FakeNode:
    def __init__(self, text, metadata=None):
        self._text = text
        # Governance metadata is required for retrieval; missing values fail
        # closed, so test stubs carry the same tenant/role a caller would.
        self.metadata = {
            "tenant_id": "tenant-demo",
            "allowed_roles": ["public"],
            **(metadata or {}),
        }

    def get_content(self):
        return self._text


class FakeHit:
    def __init__(self, node, top_dense_score=0.9):
        self.node = node
        self.top_dense_score = top_dense_score


class FakeRetriever:
    """Retriever whose .retrieve(query) always returns the same chunks."""

    def __init__(self, chunks, top_dense_score=0.9):
        self._chunks = chunks
        self._top_dense_score = top_dense_score
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return [FakeHit(c, self._top_dense_score) for c in self._chunks]


class FakeRetrieverByQuery:
    """Retriever that returns different chunks per query."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return [FakeHit(c) for c in self._mapping[query]]


def make_state(**overrides):
    state = {
        "question": "What is the capital of France?",
        "route": "",
        "sub_questions": [],
        "retrieved_chunks": [],
        "context_sufficient": None,
        "retry_count": 0,
        "answer": "",
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _reset_reranker():
    """Keep ``src.graph.nodes._reranker`` None unless a test sets it.

    ``test_eval.py`` builds the real retriever+reranker once per run; without
    this reset the cross-encoder leaks into every following graph test via the
    module-level global, and ``_rerank`` then calls ``get_content(metadata_mode=...)``
    on lightweight ``FakeNode`` stubs that do not accept that kwarg.
    """
    import src.graph.nodes as nodes

    original = nodes._reranker
    nodes._reranker = None
    yield
    nodes._reranker = original


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Where did Phileas Fogg live?", "simple"),
        ("Compare the motives of Fogg and Nemo.", "complex"),
        ("What is the difference between Ahab and Nemo?", "complex"),
        ("How do Fogg and Nemo differ?", "complex"),
        ("Fogg versus Nemo: who travelled further?", "complex"),
        ("Who is Fogg? Who is Nemo?", "complex"),
        # Titles and common words must not trigger the expensive path.
        ("Who is Mr. Darcy in Pride and Prejudice?", "simple"),
        ("What happens in War and Peace?", "simple"),
        ("Why did Nemo build the Nautilus?", "simple"),
        ("What is an indifferent narrator?", "simple"),
    ],
)
def test_route_node_uses_wording_alone(question, expected):
    with patch("src.graph.nodes.generate_answer") as llm:
        assert route_node(make_state(question=question)) == {"route": expected}
    llm.assert_not_called()


def test_decompose_node_splits_lines_and_strips_markers():
    llm = patch(
        "src.graph.nodes.generate_answer",
        return_value="What is France?\n- What is its capital?\n\n  How large is it?",
    )
    with llm:
        result = decompose_node(make_state())
    # The decomposer is capped by MAX_SUB_QUESTIONS to bound per-query work.
    assert result["sub_questions"] == [
        "What is France?",
        "What is its capital?",
    ][: settings.max_sub_questions]


def test_decompose_node_single_question():
    with patch(
        "src.graph.nodes.generate_answer", return_value="What is France?"
    ):
        result = decompose_node(make_state())
    assert result["sub_questions"] == ["What is France?"]


def test_decompose_node_normalizes_unicode_spaces_in_subquestions():
    with patch(
        "src.graph.nodes.generate_answer",
        return_value="What\u202fdrives Ahab's pursuit?\nWhat\u202fmotivates\u202fCaptain\u202fNemo?",
    ):
        result = decompose_node(make_state())
    assert result["sub_questions"] == [
        "What drives Ahab's pursuit?",
        "What motivates Captain Nemo?",
    ]


def test_decompose_node_survives_llm_outage():
    with patch(
        "src.graph.nodes.generate_answer", side_effect=LLMUnavailable("all_providers_failed")
    ):
        result = decompose_node(make_state())
    assert result == {"sub_questions": ["What is the capital of France?"]}


def test_escalate_node_decomposes_and_marks_the_route():
    with patch(
        "src.graph.nodes.generate_answer", return_value="Who is Ahab?\nWho is Nemo?"
    ):
        result = escalate_node(make_state())
    assert result == {
        "sub_questions": ["Who is Ahab?", "Who is Nemo?"],
        "route": "complex",
        "escalated": True,
    }


def test_retrieve_node_uses_question_without_subquestions():
    chunks = [
        FakeNode("alpha text", {"file_name": "a.txt"}),
        FakeNode("beta text", {"file_path": "b.md"}),
    ]
    retriever = FakeRetriever(chunks)
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(make_state(question="Where is alpha?"))
    assert result["retrieved_chunks"] == [
        {"text": "alpha text", "source": "a.txt", "query": "Where is alpha?"},
        {"text": "beta text", "source": "b.md", "query": "Where is alpha?"},
    ]
    assert retriever.queries == ["Where is alpha?"]
    assert result["evidence"] == [{"query": "Where is alpha?", "top_dense_score": 0.9}]


def test_retrieve_node_uses_subquestions():
    chunks = [FakeNode("chunk text", {"file_name": "c.md"})]
    retriever = FakeRetriever(chunks)
    state = make_state(sub_questions=["sub q1", "sub q2"])
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(state)
    assert retriever.queries == ["sub q1", "sub q2"]
    assert result["retrieved_chunks"] == [
        {"text": "chunk text", "source": "c.md", "query": "sub q1"}
    ]


def test_retrieve_node_guarantees_one_chunk_per_subquestion():
    # The second sub-question's only hit must survive even though the first
    # sub-question has more hits and could fill the budget alone.
    sub_q1 = [FakeNode(f"first {i}", {"file_name": "a.md"}) for i in range(5)]
    sub_q2 = [
        FakeNode("second only", {"file_name": "b.md"}),
        FakeNode("second extra", {"file_name": "b.md"}),
    ]
    retriever = FakeRetrieverByQuery({"q1": sub_q1, "q2": sub_q2})
    state = make_state(sub_questions=["q1", "q2"])
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(state)
    sources = {c["source"] for c in result["retrieved_chunks"]}
    queries = {c["query"] for c in result["retrieved_chunks"]}
    assert sources == {"a.md", "b.md"}
    assert queries == {"q1", "q2"}


def test_retrieve_node_tracks_query_per_chunk():
    sub_a = [FakeNode(f"ahab text {i}", {"file_name": "ahab.txt"}) for i in range(3)]
    sub_b = [FakeNode(f"nemo text {i}", {"file_name": "nemo.md"}) for i in range(3)]
    retriever = FakeRetrieverByQuery({"q1": sub_a, "q2": sub_b})
    state = make_state(sub_questions=["q1", "q2"])
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(state)
    assert any(c["query"] == "q1" for c in result["retrieved_chunks"])
    assert any(c["query"] == "q2" for c in result["retrieved_chunks"])


def test_retrieve_node_applies_reranker_when_available():
    class FakeReranker:
        def __init__(self):
            self.calls = []

        def postprocess_nodes(self, nodes, query_str):
            self.calls.append(query_str)
            return nodes

    chunks = [FakeNode(f"text {i}", {"file_name": "f.md"}) for i in range(5)]
    retriever = FakeRetriever(chunks)
    reranker = FakeReranker()
    with patch("src.graph.nodes._get_retriever", return_value=retriever), patch(
        "src.graph.nodes._reranker", reranker
    ):
        result = retrieve_node(make_state())
    assert reranker.calls == ["What is the capital of France?"]
    assert len(result["retrieved_chunks"]) == settings.top_k_reranked


def test_retrieve_node_deduplicates_repeated_text():
    chunks = [
        FakeNode("same text", {"file_name": "a.md"}),
        FakeNode("same text", {"file_name": "b.md"}),
        FakeNode("unique text", {"file_name": "c.md"}),
    ]
    retriever = FakeRetriever(chunks)
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(make_state())
    assert [c["text"] for c in result["retrieved_chunks"]] == [
        "same text",
        "unique text",
    ]


def test_retrieve_node_distributes_budget_across_subquestions():
    chunks = [
        FakeNode(f"text {i}", {"file_name": "f.md"}) for i in range(settings.top_k_reranked)
    ]
    retriever = FakeRetriever(chunks)
    state = make_state(sub_questions=["sub q1", "sub q2", "sub q3"])
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(state)
    # Every sub-question is queried and the budget is spread across all of them
    # instead of the first sub-question hogging the whole allocation.
    assert retriever.queries == ["sub q1", "sub q2", "sub q3"]
    assert len(result["retrieved_chunks"]) == settings.top_k_reranked


def test_retrieve_node_fetches_sources_from_all_subquestions():
    sub_a = [FakeNode(f"ahab text {i}", {"file_name": "ahab.txt"}) for i in range(3)]
    sub_b = [FakeNode(f"nemo text {i}", {"file_name": "nemo.md"}) for i in range(3)]
    retriever = FakeRetrieverByQuery({"q1": sub_a, "q2": sub_b})
    state = make_state(sub_questions=["q1", "q2"])
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(state)
    # A cross-book question must pull context for both halves, not just the first.
    assert retriever.queries == ["q1", "q2"]
    assert {c["source"] for c in result["retrieved_chunks"]} == {"ahab.txt", "nemo.md"}


def test_retrieve_node_widens_budget_on_retry():
    chunks = [
        FakeNode(f"text {i}", {"file_name": "f.md"})
        for i in range(settings.top_k_reranked * 2)
    ]
    retriever = FakeRetriever(chunks)
    state = make_state(sub_questions=["sub q1", "sub q2"], retry_count=1)
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(state)
    # A re-fetch doubles the total budget (5 -> 10) so the grader sees more context.
    assert len(result["retrieved_chunks"]) == settings.top_k_reranked * 2


def test_retrieve_node_drops_other_tenants():
    chunks = [
        FakeNode("mine", {"file_name": "a.md"}),
        FakeNode("theirs", {"file_name": "b.md", "tenant_id": "tenant-other"}),
    ]
    retriever = FakeRetriever(chunks)
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(make_state(tenant_id="tenant-demo", roles=["public"]))
    assert [c["text"] for c in result["retrieved_chunks"]] == ["mine"]


def test_retrieve_node_drops_chunks_without_matching_role():
    chunks = [
        FakeNode("public", {"file_name": "a.md"}),
        FakeNode("admin", {"file_name": "b.md", "allowed_roles": ["admin"]}),
        FakeNode("legacy", {"file_name": "c.md", "allowed_roles": []}),
    ]
    retriever = FakeRetriever(chunks)
    with patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = retrieve_node(make_state(tenant_id="tenant-demo", roles=["public"]))
    assert [c["text"] for c in result["retrieved_chunks"]] == ["public"]


def test_get_retriever_scopes_identity_and_caches():
    import src.graph.nodes as nodes

    visible = [FakeNode("visible", {"file_name": "a.md"})]
    hidden = [
        FakeNode("hidden", {"file_name": "b.md", "tenant_id": "tenant-other"}),
    ]
    nodes._index = MagicMock()
    nodes._index.docstore.docs = {}
    with (
        patch("src.graph.nodes.load_existing_index", return_value=nodes._index),
        patch("src.graph.nodes.load_corpus_nodes", return_value=visible + hidden),
        patch(
            "src.graph.nodes.build_hybrid_retriever", return_value="retriever"
        ) as build,
    ):
        first = nodes._get_retriever("tenant-demo", ["public"])
        second = nodes._get_retriever("tenant-demo", ["public"])
    assert first is second == "retriever"
    build.assert_called_once()
    assert build.call_args.args[1] == visible
    assert build.call_args.kwargs == {
        "tenant_id": "tenant-demo",
        "roles": ["public"],
    }
    nodes._index = None
    nodes._retrievers.clear()


def _evidence(*scores):
    return [{"query": f"q{i}", "top_dense_score": score} for i, score in enumerate(scores)]


def test_grade_node_empty_context_is_insufficient():
    result = grade_node(make_state(retrieved_chunks=[]))
    assert result == {
        "context_sufficient": False,
        "retry_count": 1,
        "grade_details": [
            {"sub_question": "What is the capital of France?", "covered": False}
        ],
    }


def test_grade_node_makes_no_llm_call():
    with patch("src.graph.nodes.generate_answer") as llm:
        grade_node(
            make_state(retrieved_chunks=[{"text": "ctx"}], evidence=_evidence(0.9))
        )
    llm.assert_not_called()


@pytest.mark.parametrize(
    "offset,expected", [(0.01, True), (0.0, True), (-0.01, False)]
)
def test_grade_node_compares_dense_score_to_the_threshold(offset, expected):
    score = settings.evidence_min_dense_score + offset
    result = grade_node(
        make_state(retrieved_chunks=[{"text": "ctx"}], evidence=_evidence(score))
    )
    assert result["context_sufficient"] is expected
    assert result["grade_details"] == [{"sub_question": "q0", "covered": expected}]


def test_grade_node_missing_score_fails_closed():
    result = grade_node(
        make_state(retrieved_chunks=[{"text": "ctx"}], evidence=_evidence(None))
    )
    assert result["context_sufficient"] is False


def test_grade_node_increments_existing_retry_count():
    result = grade_node(
        make_state(retrieved_chunks=[{"text": "ctx"}], evidence=_evidence(0.9), retry_count=3)
    )
    assert result["retry_count"] == 4


def test_grade_node_requires_every_subquestion_to_be_covered():
    # A comparison must NOT count as answerable when one half has no support.
    result = grade_node(
        make_state(retrieved_chunks=[{"text": "ctx"}], evidence=_evidence(0.9, 0.05))
    )
    assert result["context_sufficient"] is False
    assert result["grade_details"] == [
        {"sub_question": "q0", "covered": True},
        {"sub_question": "q1", "covered": False},
    ]


def test_generate_node_calls_llm_when_context_sufficient():
    chunks = [{"text": "ctx", "source": "s.md"}]
    with patch("src.graph.nodes.generate_answer", return_value="Good answer") as llm:
        result = generate_node(make_state(context_sufficient=True, retrieved_chunks=chunks))
    llm.assert_called_once_with("What is the capital of France?", chunks, plain_text=True)
    assert result == {
        "answer": "Good answer",
        "citations": [],
        "grounded": False,
        "answer_status": "ok",
        "fallback": None,
    }


def test_generate_node_uses_partial_answer_when_insufficient():
    chunks = [{"text": "ctx", "source": "s"}]
    with patch("src.graph.nodes.generate_partial_answer", return_value="Partial best-effort") as llm:
        result = generate_node(make_state(context_sufficient=False, retrieved_chunks=chunks))
    llm.assert_called_once_with(
        "What is the capital of France?", chunks, gaps=[], plain_text=True
    )
    assert result["answer"] == "Partial best-effort"
    assert result["answer_status"] == "partial"


def test_generate_node_passes_uncovered_subquestions_as_gaps():
    chunks = [{"text": "ctx", "source": "s"}]
    state = make_state(
        context_sufficient=False,
        retrieved_chunks=chunks,
        grade_details=[
            {"sub_question": "Who is Ahab?", "covered": True},
            {"sub_question": "Who is Nemo?", "covered": False},
        ],
    )
    with patch("src.graph.nodes.generate_partial_answer", return_value="Partial") as llm:
        generate_node(state)
    llm.assert_called_once_with(
        "What is the capital of France?", chunks, gaps=["Who is Nemo?"], plain_text=True
    )


@pytest.mark.parametrize("flag", [False, None])
def test_generate_node_refuses_without_chunks_or_llm_call(flag):
    with patch("src.graph.nodes.generate_partial_answer") as llm:
        result = generate_node(make_state(context_sufficient=flag))
    llm.assert_not_called()
    assert result == {
        "answer": "I don't have enough context to answer that question.",
        "citations": [],
        "grounded": False,
        "answer_status": "refused",
        "fallback": None,
    }


def test_generate_node_falls_back_to_extractive_with_valid_citations():
    chunks = [
        {
            "text": "Fogg lived at Saville Row.",
            "source": "book",
            "source_uri": "book",
            "document_id": "d1",
            "chunk_id": "c1",
        }
    ]
    with patch(
        "src.graph.nodes.generate_answer", side_effect=LLMUnavailable("all_providers_failed")
    ):
        result = generate_node(make_state(context_sufficient=True, retrieved_chunks=chunks))
    assert result["answer_status"] == "extractive"
    assert result["fallback"] == "extractive"
    assert result["grounded"] is True
    assert "[SOURCE 1] Fogg lived at Saville Row." in result["answer"]
    assert result["citations"] == [
        {"document_id": "d1", "chunk_id": "c1", "source_uri": "book"}
    ]


def test_decomposer_does_not_request_plain_text():
    # decompose_node parses the raw model string, so it must not receive a
    # normalized answer with a sources footer.
    with patch(
        "src.graph.nodes.generate_answer", return_value="Where is France?"
    ) as llm:
        decompose_node(make_state())
    assert llm.call_args.kwargs.get("plain_text") is not True


def _scripted_llm(final_answer="Generated answer", decomposition="What is France?\nWhat is its capital?"):
    """Answer by prompt type, recording each call so its cost can be asserted."""
    calls = []

    def side_effect(question, context_chunks=None, **kwargs):
        if "Break the user's question" in question:
            calls.append("decompose")
            return decomposition
        calls.append("generate")
        return final_answer

    return side_effect, calls


def _run(retriever, llm, question="What is the capital of France?"):
    with patch("src.graph.nodes.generate_answer", side_effect=llm), patch(
        "src.graph.nodes.generate_partial_answer", side_effect=llm
    ), patch("src.graph.nodes._get_retriever", return_value=retriever):
        return build_graph().invoke(make_state(question=question))


def test_build_graph_returns_compiled_app():
    app = build_graph()
    assert callable(app.invoke)


def test_full_flow_simple_route_makes_one_llm_call():
    retriever = FakeRetriever([FakeNode("contextual info", {"file_name": "doc.md"})])
    llm, calls = _scripted_llm(final_answer="The capital is Paris.")
    result = _run(retriever, llm)
    assert result["route"] == "simple"
    assert result["context_sufficient"] is True
    assert result["answer"] == "The capital is Paris."
    assert calls == ["generate"]
    assert not result.get("escalated")


def test_full_flow_weak_evidence_escalates_once_then_answers_partially():
    retriever = FakeRetriever(
        [FakeNode("thin context", {"file_name": "doc.md"})], top_dense_score=0.01
    )
    llm, calls = _scripted_llm(final_answer="Partial best-effort answer")
    result = _run(retriever, llm)
    assert result["route"] == "complex"
    assert result["escalated"] is True
    assert result["context_sufficient"] is False
    assert result["answer"] == "Partial best-effort answer"
    assert result["answer_status"] == "partial"
    assert calls == ["decompose", "generate"]
    assert retriever.queries == [
        "What is the capital of France?",
        "What is France?",
        "What is its capital?",
    ]


def test_full_flow_escalation_can_recover_full_coverage():
    class ImprovingRetriever(FakeRetriever):
        def retrieve(self, query):
            self._top_dense_score = 0.9 if query != "whole question" else 0.01
            return super().retrieve(query)

    retriever = ImprovingRetriever([FakeNode("evidence", {"file_name": "doc.md"})])
    llm, calls = _scripted_llm(final_answer="Full answer")
    result = _run(retriever, llm, question="whole question")
    assert result["context_sufficient"] is True
    assert result["answer_status"] == "ok"
    assert calls == ["decompose", "generate"]


def test_full_flow_complex_route_decomposes_up_front():
    retriever = FakeRetriever([FakeNode("contextual info", {"file_name": "doc.md"})])
    llm, calls = _scripted_llm(final_answer="It is Paris.")
    result = _run(retriever, llm, question="Compare France and Spain.")
    assert result["route"] == "complex"
    assert result["sub_questions"] == ["What is France?", "What is its capital?"]
    assert not result.get("escalated")
    assert calls == ["decompose", "generate"]


def test_full_flow_complex_weak_evidence_widens_until_the_retry_cap(monkeypatch):
    import src.graph.build_graph as graph_module

    monkeypatch.setattr(graph_module, "MAX_RETRIES", 2)
    retriever = FakeRetriever(
        [FakeNode(f"context {i}", {"file_name": "doc.md"}) for i in range(10)],
        top_dense_score=0.01,
    )
    llm, calls = _scripted_llm(final_answer="partial")
    result = _run(retriever, llm, question="Compare France and Spain.")
    assert result["retry_count"] == 2
    assert retriever.queries == ["What is France?", "What is its capital?"] * 2
    assert calls == ["decompose", "generate"]


def test_full_flow_never_exceeds_retry_cap():
    retriever = FakeRetriever(
        [FakeNode("weak context", {"file_name": "doc.md"})], top_dense_score=0.01
    )
    llm, _ = _scripted_llm(final_answer="partial best effort")
    result = _run(retriever, llm, question="Compare France and Spain.")
    assert result["retry_count"] == MAX_RETRIES
    assert result["answer"] == "partial best effort"


def test_full_flow_complex_cross_book_question_builds_context():
    sub_a = [FakeNode(f"ahab text {i}", {"file_name": "ahab.txt"}) for i in range(3)]
    sub_b = [FakeNode(f"nemo text {i}", {"file_name": "nemo.md"}) for i in range(3)]
    retriever = FakeRetrieverByQuery({"Who is Ahab?": sub_a, "Who is Nemo?": sub_b})
    llm, _ = _scripted_llm(final_answer="final answer", decomposition="Who is Ahab?\nWho is Nemo?")
    result = _run(retriever, llm, question="Compare Ahab and Nemo.")
    assert result["route"] == "complex"
    assert {c["source"] for c in result["retrieved_chunks"]} == {"ahab.txt", "nemo.md"}
    assert result["context_sufficient"] is True
    assert result["answer"] == "final answer"


def test_deadline_stops_the_graph_at_the_next_node():
    import time

    from src.resilience import DeadlineExceeded, RequestContext, request_scope

    retriever = FakeRetriever([FakeNode("text", {"file_name": "doc.md"})])
    llm, calls = _scripted_llm()
    with request_scope(RequestContext(deadline=time.monotonic() - 1)), pytest.raises(
        DeadlineExceeded
    ):
        _run(retriever, llm)
    assert retriever.queries == []
    assert calls == []


def test_request_context_reaches_nodes_through_the_graph():
    # LangGraph may run nodes on other threads; the budget must still apply.
    from src.resilience import RequestContext, current_context, request_scope

    seen = []
    retriever = FakeRetriever([FakeNode("text", {"file_name": "doc.md"})])

    def llm(question, context_chunks=None, **kwargs):
        seen.append(current_context())
        return "answer"

    context = RequestContext(max_llm_calls=1)
    with request_scope(context):
        _run(retriever, llm)
    assert seen == [context]
