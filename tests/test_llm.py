"""Tests for src/llm/langchain_llm.py prompt/context handling."""

from unittest.mock import MagicMock, patch
from src.llm.langchain_llm import (
    _format_context,
    generate_answer,
    generate_partial_answer,
)


def test_format_context_renders_source_blocks():
    chunks = [
        {"text": "alpha text", "source": "a.txt"},
        {"text": "beta text", "source": "b.md"},
    ]
    result = _format_context(chunks)
    assert "[SOURCE 1] alpha text" in result
    assert "Source: a.txt" in result
    assert "[SOURCE 2] beta text" in result
    assert "Source: b.md" in result


def test_format_context_empty_when_no_chunks():
    assert _format_context(None) == ""
    assert _format_context([]) == ""


def test_generate_answer_passes_formatted_context():
    chain = MagicMock()
    chain.invoke.return_value = "Answered."
    chunks = [{"text": "alpha text", "source": "a.txt"}]
    with patch("src.llm.langchain_llm.build_rag_chain", return_value=chain):
        answer = generate_answer("question?", chunks)
    assert answer == "Answered."
    inputs = chain.invoke.call_args.args[0]
    assert inputs["question"] == "question?"
    assert "[SOURCE 1] alpha text" in inputs["context"]


def test_generate_answer_passes_empty_context_when_none():
    chain = MagicMock()
    chain.invoke.return_value = "Direct answer."
    with patch("src.llm.langchain_llm.build_rag_chain", return_value=chain):
        answer = generate_answer("hi", None)
    assert answer == "Direct answer."
    inputs = chain.invoke.call_args.args[0]
    assert inputs["context"] == ""


def test_generate_partial_answer_uses_its_own_chain():
    chain = MagicMock()
    chain.invoke.return_value = "Partial best-effort."
    chunks = [{"text": "alpha text", "source": "a.txt"}]
    with patch("src.llm.langchain_llm.build_partial_answer_chain", return_value=chain):
        answer = generate_partial_answer("question?", chunks)
    assert answer == "Partial best-effort."
    inputs = chain.invoke.call_args.args[0]
    assert inputs["question"] == "question?"
    assert "[SOURCE 1] alpha text" in inputs["context"]