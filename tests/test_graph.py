"""Tests for the node functions in src/graph/nodes.py and src/graph/build_graph.py."""

from unittest.mock import MagicMock, patch
import pytest
from src.config import settings
from src.graph.build_graph import build_graph
from src.graph.nodes import (
    MAX_RETRIES,
    decompose_node,
    generate_node,
    grade_node,
    no_retrieval_node,
    retrieve_node,
    route_node,
)


class FakeNode:
    def __init__(self, text, metadata=None):
        self._text = text
        self.metadata = metadata or {}

    def get_content(self):
        return self._text


class FakeHit:
    def __init__(self, node):
        self.node = node


class FakeRetriever:
    """Retriever whose .retrieve(query) always returns the same chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return [FakeHit(c) for c in self._chunks]


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
    "llm_output,expected",
    [
        ("no_retrieval", "no_retrieval"),
        ("simple", "simple"),
        ("complex", "complex"),
        ("  Simple\n", "simple"),
        ("COMPLEX", "complex"),
    ],
)
def test_route_node_maps_llm_output(llm_output, expected):
    with patch("src.graph.nodes.generate_answer", return_value=llm_output):
        assert route_node(make_state()) == {"route": expected}


def test_route_node_falls_back_to_simple_on_unexpected_output():
    with patch("src.graph.nodes.generate_answer", return_value="I am confused"):
        assert route_node(make_state()) == {"route": "simple"}


def test_decompose_node_splits_lines_and_strips_markers():
    llm = patch(
        "src.graph.nodes.generate_answer",
        return_value="What is France?\n- What is its capital?\n\n  How large is it?",
    )
    with llm:
        result = decompose_node(make_state())
    assert result["sub_questions"] == [
        "What is France?",
        "What is its capital?",
        "How large is it?",
    ]


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
    assert len(result["retrieved_chunks"]) == len(chunks)


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


def test_grade_node_empty_context_is_insufficient():
    result = grade_node(make_state(retrieved_chunks=[]))
    assert result == {
        "context_sufficient": False,
        "retry_count": 1,
        "grade_details": [
            {"sub_question": "What is the capital of France?", "covered": False}
        ],
    }


def test_grade_node_yes_means_sufficient():
    with patch("src.graph.nodes.generate_answer", return_value="YES"):
        result = grade_node(make_state(retrieved_chunks=[{"text": "ctx", "source": "s"}]))
    assert result == {
        "context_sufficient": True,
        "retry_count": 1,
        "grade_details": [
            {"sub_question": "What is the capital of France?", "covered": True}
        ],
    }


def test_grade_node_no_means_insufficient():
    with patch("src.graph.nodes.generate_answer", return_value="NO"):
        result = grade_node(make_state(retrieved_chunks=[{"text": "ctx", "source": "s"}]))
    assert result["context_sufficient"] is False
    assert result["grade_details"] == [
        {"sub_question": "What is the capital of France?", "covered": False}
    ]


def test_grade_node_accepts_leading_whitespace_and_yes_prefix():
    with patch("src.graph.nodes.generate_answer", return_value="  YES, this covers it"):
        result = grade_node(make_state(retrieved_chunks=[{"text": "ctx", "source": "s"}]))
    assert result["context_sufficient"] is True


def test_grade_node_increments_existing_retry_count():
    with patch("src.graph.nodes.generate_answer", return_value="YES"):
        result = grade_node(
            make_state(retrieved_chunks=[{"text": "ctx", "source": "s"}], retry_count=3)
        )
    assert result == {
        "context_sufficient": True,
        "retry_count": 4,
        "grade_details": [
            {"sub_question": "What is the capital of France?", "covered": True}
        ],
    }


def test_grade_node_complex_passes_when_all_subquestions_covered():
    state = make_state(
        route="complex",
        sub_questions=["Who is Ahab?", "Who is Nemo?"],
        retrieved_chunks=[{"text": "ctx", "source": "s"}],
    )
    with patch("src.graph.nodes.generate_answer", side_effect=["YES", "YES"]):
        result = grade_node(state)
    assert result["context_sufficient"] is True
    assert result["grade_details"] == [
        {"sub_question": "Who is Ahab?", "covered": True},
        {"sub_question": "Who is Nemo?", "covered": True},
    ]


def test_grade_node_complex_insufficient_with_partial_coverage():
    # A comparison question must NOT be declared fully answerable when half
    # (e.g. the Nemo side) is uncovered; partial coverage must re-retrieve.
    state = make_state(
        route="complex",
        sub_questions=["Who is Ahab?", "Who is Nemo?"],
        retrieved_chunks=[{"text": "ctx", "source": "s"}],
    )
    with patch("src.graph.nodes.generate_answer", side_effect=["YES", "NO"]):
        result = grade_node(state)
    assert result["context_sufficient"] is False
    assert result["grade_details"] == [
        {"sub_question": "Who is Ahab?", "covered": True},
        {"sub_question": "Who is Nemo?", "covered": False},
    ]


def test_grade_node_complex_insufficient_when_no_subquestion_covered():
    state = make_state(
        route="complex",
        sub_questions=["Who is Ahab?", "Who is Nemo?"],
        retrieved_chunks=[{"text": "ctx", "source": "s"}],
    )
    with patch("src.graph.nodes.generate_answer", side_effect=["NO", "NO"]):
        result = grade_node(state)
    assert result["context_sufficient"] is False



def test_generate_node_calls_llm_when_context_sufficient():
    chunks = [{"text": "ctx", "source": "s.md"}]
    with patch("src.graph.nodes.generate_answer", return_value="Good answer") as llm:
        result = generate_node(make_state(context_sufficient=True, retrieved_chunks=chunks))
    llm.assert_called_once_with("What is the capital of France?", chunks)
    assert result == {"answer": "Good answer"}


def test_generate_node_uses_partial_answer_when_insufficient():
    chunks = [{"text": "ctx", "source": "s"}]
    with patch("src.graph.nodes.generate_partial_answer", return_value="Partial best-effort") as llm:
        result = generate_node(make_state(context_sufficient=False, retrieved_chunks=chunks))
    llm.assert_called_once_with("What is the capital of France?", chunks, gaps=[])
    assert result == {"answer": "Partial best-effort"}


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
        result = generate_node(state)
    llm.assert_called_once_with(
        "What is the capital of France?", chunks, gaps=["Who is Nemo?"]
    )
    assert result == {"answer": "Partial"}


def test_generate_node_does_not_call_llm_when_insufficient_and_no_chunks():
    with patch("src.graph.nodes.generate_partial_answer") as llm:
        result = generate_node(make_state(context_sufficient=False))
    llm.assert_not_called()
    assert result == {"answer": "I don't have enough context to answer that question."}


def test_generate_node_defaults_to_dont_know_when_flag_missing():
    with patch("src.graph.nodes.generate_partial_answer") as llm:
        result = generate_node(make_state(context_sufficient=None))
    llm.assert_not_called()
    assert result == {"answer": "I don't have enough context to answer that question."}



def test_no_retrieval_node_answers_without_context():
    with patch("src.graph.nodes.generate_answer", return_value="Paris is the capital.") as llm:
        result = no_retrieval_node(make_state())
    llm.assert_called_once_with("What is the capital of France?", None)
    assert result == {"answer": "Paris is the capital."}


def _route_llm(routing_response, grade_response="YES", final_answer="Generated answer"):
    """Generate a side_effect that pretends each prompt type has a fixed reply."""

    def side_effect(question, context_chunks=None, **kwargs):
        if "Classify the following question" in question:
            return routing_response
        if "exactly one word: YES or NO" in question:
            return grade_response
        return final_answer

    return side_effect


def test_build_graph_returns_compiled_app():
    app = build_graph()
    assert callable(app.invoke)


def test_full_flow_no_retrieval_skips_retrieval():
    llm = _route_llm("no_retrieval", final_answer="Paris is the capital.")
    with (
        patch("src.graph.nodes.generate_answer", side_effect=llm),
        patch("src.graph.nodes._get_retriever") as retriever,
    ):
        result = build_graph().invoke(make_state())
    retriever.assert_not_called()
    assert result["route"] == "no_retrieval"
    assert result["answer"] == "Paris is the capital."


def test_full_flow_simple_route():
    retriever = FakeRetriever([FakeNode("contextual info", {"file_name": "doc.md"})])
    llm = _route_llm("simple", grade_response="YES", final_answer="The capital is Paris.")
    with patch("src.graph.nodes.generate_answer", side_effect=llm), patch(
        "src.graph.nodes._get_retriever", return_value=retriever
    ):
        result = build_graph().invoke(make_state())
    assert result["route"] == "simple"
    assert result["context_sufficient"] is True
    assert result["answer"] == "The capital is Paris."


def test_full_flow_simple_insufficient_context_ends_with_partial_answer():
    retriever = FakeRetriever([FakeNode("thin context", {"file_name": "doc.md"})])
    llm = _route_llm("simple", grade_response="NO", final_answer="Partial best-effort answer")
    with patch("src.graph.nodes.generate_answer", side_effect=llm), patch(
        "src.graph.nodes.generate_partial_answer", side_effect=llm
    ), patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = build_graph().invoke(make_state())
    assert result["route"] == "simple"
    assert result["context_sufficient"] is False
    assert result["retry_count"] == MAX_RETRIES
    assert result["answer"] == "Partial best-effort answer"


def test_full_flow_complex_route_decomposes_then_retries_then_answers():
    retriever = FakeRetriever([FakeNode("contextual info", {"file_name": "doc.md"})])

    def llm(question, context_chunks=None, **kwargs):
        if "Classify the following question" in question:
            return "complex"
        if "Break the user's question" in question:
            return "What is France?\nWhat is its capital?"
        if "exactly one word: YES or NO" in question:
            return "NO"
        return "It is Paris."

    with patch("src.graph.nodes.generate_answer", side_effect=llm), patch(
        "src.graph.nodes.generate_partial_answer", side_effect=llm
    ), patch("src.graph.nodes._get_retriever", return_value=retriever):
        result = build_graph().invoke(make_state())
    assert result["route"] == "complex"
    assert result["sub_questions"] == ["What is France?", "What is its capital?"]
    # Complex now re-retrieves (instead of immediately forcing "don't know"),
    # so every sub-question is fetched on the first pass plus subsequent retries.
    assert retriever.queries == ["What is France?", "What is its capital?"] * MAX_RETRIES
    assert result["context_sufficient"] is False
    assert result["retry_count"] == MAX_RETRIES
    # With retries exhausted, generate emits a partial best-effort answer.
    assert result["answer"] == "It is Paris."


def test_full_flow_complex_cross_book_question_builds_context():
    sub_a = [FakeNode(f"ahab text {i}", {"file_name": "ahab.txt"}) for i in range(3)]
    sub_b = [FakeNode(f"nemo text {i}", {"file_name": "nemo.md"}) for i in range(3)]
    retriever = FakeRetrieverByQuery({"Who is Ahab?": sub_a, "Who is Nemo?": sub_b})

    def llm(question, context_chunks=None, **kwargs):
        if "Classify the following question" in question:
            return "complex"
        if "Break the user's question" in question:
            return "Who is Ahab?\nWho is Nemo?"
        if "exactly one word: YES or NO" in question:
            return "YES"
        return "final answer"

    with patch("src.graph.nodes.generate_answer", side_effect=llm), patch(
        "src.graph.nodes._get_retriever", return_value=retriever
    ):
        result = build_graph().invoke(make_state())
    assert result["route"] == "complex"
    assert {c["source"] for c in result["retrieved_chunks"]} == {"ahab.txt", "nemo.md"}
    assert result["context_sufficient"] is True
    assert result["answer"] == "final answer"


def test_full_flow_never_exceeds_retry_cap():
    retriever = FakeRetriever([FakeNode("weak context", {"file_name": "doc.md"})])
    llm = _route_llm("simple", grade_response="NO", final_answer="partial best effort")
    for _ in range(MAX_RETRIES + 1):
        with patch("src.graph.nodes.generate_answer", side_effect=llm), patch(
            "src.graph.nodes.generate_partial_answer", side_effect=llm
        ), patch("src.graph.nodes._get_retriever", return_value=retriever):
            result = build_graph().invoke(make_state())
    assert result["retry_count"] == MAX_RETRIES
    assert result["answer"] == "partial best effort"