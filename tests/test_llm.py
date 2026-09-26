"""Tests for src/llm/langchain_llm.py prompt/context handling."""

from unittest.mock import MagicMock, patch
from src.llm.langchain_llm import (
    _append_sources,
    _format_context,
    _normalize_output,
    cited_source_indices,
    display_source_name,
    generate_answer,
    generate_partial_answer,
    normalize_chunk_text,
    split_sources_footer,
)

FOGG_SLUG = "Around_the_World_in_Eighty_Days_-_Jules_Verne"


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


def test_format_context_renders_readable_source_name():
    chunks = [{"text": "fogg text", "source": FOGG_SLUG}]
    result = _format_context(chunks)
    assert "Source: Around the World in Eighty Days - Jules Verne" in result


def test_display_source_name_splits_slug_into_title_and_author():
    assert display_source_name(FOGG_SLUG) == "Around the World in Eighty Days - Jules Verne"
    assert (
        display_source_name("Alice's_Adventures_in_Wonderland_-_Lewis_Carroll")
        == "Alice's Adventures in Wonderland - Lewis Carroll"
    )


def test_display_source_name_passes_through_unseparated_source():
    assert display_source_name("a.txt") == "a.txt"
    assert display_source_name("") == ""


def test_normalize_output_strips_markdown_markers():
    assert _normalize_output("was **Number 7** here") == "was Number 7 here"
    assert _normalize_output("a *small* claim") == "a small claim"
    assert _normalize_output("a _small_ claim") == "a small claim"
    assert _normalize_output("see `No. 7` below") == "see No. 7 below"


def test_normalize_output_drops_heading_and_bullet_markers():
    assert _normalize_output("## Details") == "Details"
    assert _normalize_output("### Heading 2") == "Heading 2"
    assert _normalize_output("- first item") == "first item"
    assert _normalize_output("* second item") == "second item"
    assert _normalize_output("- **bold item**") == "bold item"


def test_normalize_output_keeps_lookalike_hashes_and_underscores():
    assert _normalize_output("#1 not a heading") == "#1 not a heading"
    assert _normalize_output("#hashtag") == "#hashtag"
    assert _normalize_output("1. not a bullet") == "1. not a bullet"
    assert _normalize_output("-no space") == "-no space"
    assert _normalize_output(f"the slug {FOGG_SLUG} survives") == (
        f"the slug {FOGG_SLUG} survives"
    )
    assert _normalize_output("read file_name and snake_case") == (
        "read file_name and snake_case"
    )


def test_normalize_output_folds_unicode_punctuation():
    assert _normalize_output("Fogg\u2019s row") == "Fogg's row"
    assert _normalize_output("the \u201cCity\u201d") == 'the "City"'
    assert _normalize_output("\u3010SOURCE 1\u3011") == "[SOURCE 1]"
    assert _normalize_output("Byron\u2014at least") == "Byron-at least"
    assert _normalize_output("No.\u202f7") == "No. 7"
    assert _normalize_output("Paris\u2011London") == "Paris-London"


def test_normalize_output_collapses_blank_lines_and_trims():
    assert _normalize_output("a\n\n\n\n\nb") == "a\n\nb"
    assert _normalize_output("\n\n  answer  \n\n") == "answer"


def test_normalize_output_handles_empty_text():
    assert _normalize_output("") == ""


def test_normalize_chunk_text_folds_typographic_punctuation():
    assert normalize_chunk_text("Fogg\u2019s row") == "Fogg's row"
    assert normalize_chunk_text("the \u201cCity\u201d") == 'the "City"'
    assert normalize_chunk_text("Paris\u2011London") == "Paris-London"
    assert normalize_chunk_text("Byron\u2014at least") == "Byron-at least"
    assert normalize_chunk_text("No.\u202f7") == "No. 7"
    assert normalize_chunk_text("a\u00a0b") == "a b"


def test_normalize_chunk_text_keeps_markdown_markers():
    # The chunk profile stops before the Markdown regexes on purpose. Prose
    # legitimately contains these and the corpus has no Markdown to strip, so
    # stripping it here can only lose text: a leading "- " and any span between
    # two asterisks would be eaten.
    assert normalize_chunk_text("was **Number 7** here") == "was **Number 7** here"
    assert normalize_chunk_text("a *small* claim") == "a *small* claim"
    assert normalize_chunk_text("- first item") == "- first item"
    assert normalize_chunk_text("## Details") == "## Details"
    assert normalize_chunk_text("see `No. 7`") == "see `No. 7`"


def test_normalize_chunk_text_keeps_soft_hyphens_and_zero_width():
    # Data-faithful on purpose: the renderer hides these, normalization must not
    # drop them, so nothing rewrites the stored text.
    assert normalize_chunk_text("man\u00adscript") == "man\u00adscript"
    assert normalize_chunk_text("thin\u200bspace") == "thin\u200bspace"
    assert normalize_chunk_text("") == ""


def test_split_sources_footer_strips_only_the_footer():
    answer = (
        "Fogg lived at Number 7 [SOURCE 1].\n\nSources:\n"
        "  [1] Around the World in Eighty Days - Jules Verne"
    )
    assert split_sources_footer(answer) == "Fogg lived at Number 7 [SOURCE 1]."
    assert split_sources_footer("No footer here.") == "No footer here."
    assert split_sources_footer("") == ""


def test_cited_source_indices_returns_raw_matches_without_fallback():
    chunks = [{"text": "a"}, {"text": "b"}]
    assert cited_source_indices("No citation at all.", chunks) == []
    assert cited_source_indices("Both [SOURCE 2] and [SOURCE 1].", chunks) == [2, 1]
    assert cited_source_indices("Repeated [SOURCE 1] twice [SOURCE 1].", chunks) == [1]


def test_cited_source_indices_ignores_out_of_range_and_absent_chunks():
    assert cited_source_indices("Claim [SOURCE 9].", [{"text": "a"}]) == []
    assert cited_source_indices("Claim [SOURCE 1].", None) == []
    assert cited_source_indices("Claim [SOURCE 1].", []) == []


def test_append_sources_names_cited_book():
    answer = "Fogg lived at Number 7 [SOURCE 1]."
    result = _append_sources(answer, [{"text": "t", "source": FOGG_SLUG}])
    assert result == (
        "Fogg lived at Number 7 [SOURCE 1].\n\nSources:\n"
        "  [1] Around the World in Eighty Days - Jules Verne"
    )


def test_append_sources_resolves_lenticular_bracket_markers():
    # The model emits U+3010/U+3011 rather than "[SOURCE 1]"; an ASCII-only
    # regex silently finds nothing here and the footer would list every chunk.
    answer = "Fogg lived at Number 7 \u3010SOURCE 1\u3011."
    result = _append_sources(
        _normalize_output(answer), [{"text": "t", "source": FOGG_SLUG}]
    )
    assert "[SOURCE 1]." in result
    assert "Sources:\n  [1] Around the World in Eighty Days - Jules Verne" in result


def test_append_sources_dedupes_repeated_book():
    chunks = [{"text": f"t{i}", "source": FOGG_SLUG} for i in range(5)]
    result = _append_sources("He lived there [SOURCE 1].", chunks)
    assert result.count("Around the World in Eighty Days - Jules Verne") == 1


def test_append_sources_orders_by_first_citation():
    chunks = [
        {"text": "a", "source": "Moby_Dick_-_Herman_Melville"},
        {"text": "b", "source": FOGG_SLUG},
    ]
    result = _append_sources("Two facts [SOURCE 2] and [SOURCE 1].", chunks)
    assert result.index("Around the World") < result.index("Moby Dick")


def test_append_sources_lists_all_chunks_when_nothing_cited():
    chunks = [
        {"text": "a", "source": FOGG_SLUG},
        {"text": "b", "source": "Moby_Dick_-_Herman_Melville"},
    ]
    result = _append_sources("No citation at all.", chunks)
    assert "  [1] Around the World in Eighty Days - Jules Verne" in result
    assert "  [2] Moby Dick - Herman Melville" in result


def test_append_sources_ignores_out_of_range_markers():
    chunks = [{"text": "a", "source": FOGG_SLUG}]
    result = _append_sources("Claim [SOURCE 9].", chunks)
    assert "Sources:\n  [1] Around the World in Eighty Days - Jules Verne" in result
    assert "[SOURCE 9]" in result


def test_append_sources_without_chunks_leaves_text_untouched():
    assert _append_sources("Direct answer.", None) == "Direct answer."
    assert _append_sources("Direct answer [SOURCE 1].", []) == "Direct answer [SOURCE 1]."


def test_generate_answer_without_plain_text_returns_raw_model_output():
    raw = "Fogg\u2019s home was **Number 7** \u3010SOURCE 1\u3011."
    chain = MagicMock()
    chain.invoke.return_value = raw
    with patch("src.llm.langchain_llm.build_rag_chain", return_value=chain):
        answer = generate_answer(
            "where?", [{"text": "t", "source": FOGG_SLUG}], plain_text=False
        )
    assert answer == raw


def test_generate_answer_plain_text_normalizes_and_names_sources():
    chain = MagicMock()
    chain.invoke.return_value = (
        "Fogg\u2019s home was **Number 7, Saville Row** "
        "\u3010SOURCE 1\u3011."
    )
    with patch("src.llm.langchain_llm.build_rag_chain", return_value=chain):
        answer = generate_answer(
            "where?", [{"text": "t", "source": FOGG_SLUG}], plain_text=True
        )
    assert answer == (
        "Fogg's home was Number 7, Saville Row [SOURCE 1].\n\nSources:\n"
        "  [1] Around the World in Eighty Days - Jules Verne"
    )
    assert "**" not in answer


def test_generate_answer_plain_text_with_no_context_adds_no_footer():
    chain = MagicMock()
    chain.invoke.return_value = "## Paris\n- is the capital"
    with patch("src.llm.langchain_llm.build_rag_chain", return_value=chain):
        answer = generate_answer("capital?", None, plain_text=True)
    assert answer == "Paris\nis the capital"


def test_generate_partial_answer_plain_text_normalizes_and_names_sources():
    chain = MagicMock()
    chain.invoke.return_value = "He lived at **Number 7** [SOURCE 1]."
    with patch("src.llm.langchain_llm.build_partial_answer_chain", return_value=chain):
        answer = generate_partial_answer(
            "where?", [{"text": "t", "source": FOGG_SLUG}], plain_text=True
        )
    assert answer == (
        "He lived at Number 7 [SOURCE 1].\n\nSources:\n"
        "  [1] Around the World in Eighty Days - Jules Verne"
    )


def test_generate_partial_answer_without_plain_text_returns_raw_model_output():
    raw = "He lived at **Number 7** [SOURCE 1]."
    chain = MagicMock()
    chain.invoke.return_value = raw
    with patch("src.llm.langchain_llm.build_partial_answer_chain", return_value=chain):
        answer = generate_partial_answer(
            "where?", [{"text": "t", "source": FOGG_SLUG}]
        )
    assert answer == raw
