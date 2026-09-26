"""
Wrap the hosted LLM using LangChain's ChatGroq + LCEL prompt composition.
"""

import re
import ssl
import time
import unicodedata
from functools import lru_cache
import httpx
import truststore
from groq import RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from src.config import settings

MAX_RATE_LIMIT_RETRIES = 6
_RETRY_AFTER_RE = re.compile(r"Please try again in ([\d.]+)s")

_SQUOTE = "'"
_DQUOTE = '"'

# NFKC folds the exotic spaces and full-width forms but leaves curly quotes and
# CJK lenticular brackets alone, and Groq emits both: answers arrive carrying a
# U+2019 apostrophe and U+3010 SOURCE U+3011 instead of "[SOURCE 1]". Without
# this table the markers survive normalization and no citation regex can see
# them, so the footer silently falls back to listing every retrieved chunk.
_PUNCTUATION_TRANSLATION = {
    0x3010: "[",
    0x3011: "]",
    0x2018: _SQUOTE,
    0x2019: _SQUOTE,
    0x201A: _SQUOTE,
    0x201B: _SQUOTE,
    0x2039: _SQUOTE,
    0x203A: _SQUOTE,
    0x201C: _DQUOTE,
    0x201D: _DQUOTE,
    0x201E: _DQUOTE,
    0x201F: _DQUOTE,
    0x00AB: _DQUOTE,
    0x00BB: _DQUOTE,
    0x2010: "-",
    0x2012: "-",
    0x2013: "-",
    0x2014: "-",
    0x2212: "-",
}

_CODE_SPAN_RE = re.compile(r"`+([^`\n]+)`+")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")

# Every Markdown pattern below is deliberately narrower than a plain
# ``str.replace``. The underscore and star emphasis patterns are fenced by
# lookarounds so identifiers survive: the source slugs this module has to parse
# are "Around_the_World_-_Jules_Verne", and an unguarded "_" strip would
# destroy the very string the citation footer is built from. The heading pattern
# requires whitespace after the hashes so "#1" and "#hashtag" are left alone.
_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<![\w_])_([^_\n]+)_(?![\w_])")
_HEADING_RE = re.compile(r"(?:\A|(?<=\n))[ \t]{0,3}#{1,6}(?=[ \t])[ \t]*")
_BULLET_RE = re.compile(r"(?:\A|(?<=\n))[ \t]*[-*+][ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Lenient because the model improvises the citation syntax despite the prompt
# asking for "[SOURCE n]"; both square brackets are optional so a bare
# "SOURCE 3" still resolves.
_CITATION_RE = re.compile(r"\[?\s*SOURCE\s*#?\s*(\d+)\s*\]?", re.IGNORECASE)

# The exact separator ``_append_sources`` writes between an answer and its
# footer. Both the footer builder and ``split_sources_footer`` use it so the
# boundary is never spelled twice.
_SOURCES_FOOTER = "\n\nSources:\n"

RAG_SYSTEM_PROMPT = """You are a helpful assistant.
- When a context block is provided, answer using ONLY that context.
- Cite the relevant sources as [SOURCE n] for every factual claim you make,
  reusing the exact label of the context block the claim came from.
- Write plain text: no Markdown, no bold or italic markers, no headings and no
  bullet symbols.
- Every claim must be explicitly supported by the provided text. Do NOT use
  outside knowledge of the books or fill in gaps with details you happen to
  remember, even if they are accurate.
- If the context is insufficient for part of the question, say explicitly what
  you could not answer instead of guessing.
- When no context is provided, answer the question directly.
"""

PARTIAL_ANSWER_SYSTEM_PROMPT = """You are a helpful assistant.
Answer the question as well as you can using ONLY the provided context. The
context may be incomplete: answer only what the context supports and explicitly
note what you could not answer because the context is missing. Do not guess,
and do not use outside knowledge of the books to fill in gaps, even if you
happen to know the correct details.
Write plain text with no Markdown, and cite each context block you rely on by
its exact [SOURCE n] label.
{extra_instructions}
"""


@lru_cache(maxsize=2)
def _get_chat_model(model: str) -> ChatGroq:
    """Build a ChatGroq client for ``model`` once and reuse it.

    Windows enterprise proxies often use a root CA that is trusted by the
    operating system but not by certifi, which httpx uses by default, so the
    client is wired to a truststore-backed SSL context.

    Args:
        model: The Groq model identifier to target.

    Returns:
        A cached ChatGroq client bound to ``model``.
    """
    # Windows enterprise proxies often use a root CA that is trusted by the
    # operating system but not by certifi, which httpx uses by default.
    http_client = httpx.Client(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    )
    return ChatGroq(
        model=model,
        temperature=settings.groq_temperature,
        model_kwargs={"seed": settings.groq_seed},
        http_client=http_client,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    """Build the production ChatGroq client once per process and reuse it.

    Returns:
        A cached ChatGroq client using the configured ``groq_model``.
    """
    return _get_chat_model(settings.groq_model)


@lru_cache(maxsize=1)
def get_judge_chat_model() -> ChatGroq:
    """Build the RAGAS judge ChatGroq client once per process and reuse it.

    Returns:
        A cached ChatGroq client using the configured ``ragas_judge_model``.
    """
    return _get_chat_model(settings.ragas_judge_model)


@lru_cache(maxsize=1)
def build_rag_chain():
    """Compose prompt | model | parser once per process and reuse it.

    Chains the RAG system prompt and context/question template through the
    production chat model and a plain string output parser. The result is
    cached so ``generate_answer`` never recompiles the chain.

    Returns:
        The composed LCEL chain ``template | chat_model | parser``.
    """

    template = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    parser = StrOutputParser()

    return template | get_chat_model() | parser


@lru_cache(maxsize=1)
def build_partial_answer_chain():
    """Compose the partial-answer prompt | model | parser once per process.

    The chain answers from whatever context is available while explicitly
    noting gaps, used by ``generate_partial_answer`` when grading judged the
    retrieved context insufficient. Cached like ``build_rag_chain``.

    Returns:
        The composed LCEL chain ``template | chat_model | parser``.
    """

    template = ChatPromptTemplate.from_messages([
        ("system", PARTIAL_ANSWER_SYSTEM_PROMPT.format(extra_instructions="")),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{note}"),
    ])

    parser = StrOutputParser()

    return template | get_chat_model() | parser


def display_source_name(source: str) -> str:
    """Render a stored source identifier as a readable title and author.

    The index stores each book under its ``"Book Name"`` slug, which joins the
    title and author with ``"_-_"`` and underscores every word
    (``Around_the_World_in_Eighty_Days_-_Jules_Verne``). Prompt and footer both
    show the readable form so the model attributes a claim to a book and the
    user sees one, while the slug stays intact in ``retrieved_chunks`` for API
    clients.

    Args:
        source: The stored identifier, or an empty string when the retriever
            supplied no filename.

    Returns:
        ``"Title - Author"`` for a slug, or ``source`` unchanged when it has no
        ``"_-_"`` separator (a plain filename such as ``a.txt``).
    """
    title, separator, author = (source or "").partition("_-_")
    if not separator:
        return source or ""
    return f"{title.replace('_', ' ')} - {author.replace('_', ' ')}"


def _normalize_output(text: str) -> str:
    """Flatten model output into plain text suitable for a terminal.

    The model answers in Markdown and with typographic punctuation, so a raw
    answer reaches the CLI carrying bold markers, curly quotes and a stray
    non-breaking space or two. Normalization is applied after the brackets are
    folded, because the citation regex only recognizes ASCII ``[SOURCE n]``.

    Args:
        text: The raw model answer.

    Returns:
        The answer with Markdown markers removed, punctuation folded to ASCII,
        bullets and heading markers dropped, and runs of blank lines collapsed.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_PUNCTUATION_TRANSLATION)
    normalized = _CODE_SPAN_RE.sub(r"\1", normalized)
    normalized = _BOLD_RE.sub(r"\1", normalized)
    normalized = _ITALIC_STAR_RE.sub(r"\1", normalized)
    normalized = _ITALIC_UNDERSCORE_RE.sub(r"\1", normalized)
    normalized = _HEADING_RE.sub("", normalized)
    normalized = _BULLET_RE.sub("", normalized)
    normalized = _BLANK_LINES_RE.sub("\n\n", normalized)
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def normalize_chunk_text(text: str) -> str:
    """Fold a retrieved chunk's typographic punctuation to ASCII, nothing else.

    The corpus is typographic at the source rather than at the model: 92.3% of
    stored chunks carry a curly quote or a dash that NFKC alone leaves untouched,
    so a raw chunk printed next to an answer that ``_normalize_output`` has
    folded to ASCII disagrees with it typographically. Running both through the
    same two steps makes a source dump readable beside the answer it supports.

    Deliberately not ``_normalize_output``. That function's Markdown regexes are
    tuned for model output and are a net loss on prose: ``_BULLET_RE`` eats a
    leading "- " and ``_ITALIC_STAR_RE`` eats anything between two asterisks,
    which several Gutenberg editions use for italics and footnote marks. The
    corpus carries essentially no Markdown to remove, so stripping it buys
    nothing and can only lose text.

    Lenient also means it rewrites no whitespace and drops no characters. Soft
    hyphens and zero-width characters survive, because altering stored text is a
    data change, not a rendering one; display code strips them at display time.

    Args:
        text: Chunk text exactly as stored in the index.

    Returns:
        The text with compatibility forms and typographic punctuation folded to
        ASCII, otherwise identical to the input.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).translate(_PUNCTUATION_TRANSLATION)


def split_sources_footer(text: str) -> str:
    """Return ``text`` with any ``Sources:`` footer removed.

    An answer's footer is built from bracketed numbers (``[1] Title - Author``)
    that are not citations, so anything scanning an answer for citation markers
    has to look at the body alone or it will read the footer as its own
    evidence.

    Args:
        text: A finalized answer, with or without a footer.

    Returns:
        The answer body, or ``""`` when ``text`` is empty.
    """
    return (text or "").split(_SOURCES_FOOTER, 1)[0]


def cited_source_indices(text: str, context_chunks: list[dict] | None) -> list[int]:
    """Return the context block numbers ``text`` cites, in first-citation order.

    This is the single place the lenient ``_CITATION_RE`` is applied, so the
    answer footer and the source dump can never disagree about what was cited.
    The footer's "cite nothing, list everything" fallback is applied by
    ``_append_sources`` and not here: an empty return means the answer really
    did cite nothing, which is exactly the signal the dump needs in order to
    label every block "not cited".

    Args:
        text: Answer text to scan. Pass the body only, via
            ``split_sources_footer``, so the footer's own numbers are not read
            as citations.
        context_chunks: The blocks the answer was generated from, numbered from
            1. Markers outside that range are ignored.

    Returns:
        Distinct indices in order of first appearance; empty when nothing is
        cited or no chunks were provided.
    """
    if not context_chunks:
        return []
    cited: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        index = int(match.group(1))
        if 1 <= index <= len(context_chunks) and index not in cited:
            cited.append(index)
    return cited


def _append_sources(text: str, context_chunks: list[dict] | None) -> str:
    """Append a ``Sources:`` footer naming every book the answer cites.

    Names come from the retrieved chunks rather than from the model's prose, so
    a title can never be garbled. Entries are deduplicated by name because a
    single book often fills the whole retrieval budget, and ordered by first
    citation so the footer reads in the order the claims appear.

    The citation scan is delegated to ``cited_source_indices``; only the
    fallback differs. A model that cites nothing still gets a footer listing
    every block, because a wrong-looking footer is worse than a generous one.
    Callers that need to tell "cited nothing" apart from "cited everything" use
    ``cited_source_indices`` directly.

    Args:
        text: A normalized answer.
        context_chunks: The chunks that were rendered into the prompt, numbered
            from 1 in the same order the answer cites them.

    Returns:
        ``text`` with a footer appended, or unchanged when no chunks were
        provided or none of them carry a name.
    """
    if not context_chunks:
        return text

    cited = cited_source_indices(text, context_chunks)
    if not cited:
        cited = list(range(1, len(context_chunks) + 1))

    lines: list[str] = []
    seen: set[str] = set()
    for index in cited:
        chunk = context_chunks[index - 1]
        raw = chunk.get("source", "") if isinstance(chunk, dict) else str(chunk)
        name = display_source_name(raw)
        label = name or f"chunk {index}"
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"  [{index}] {label}")
    if not lines:
        return text
    return f"{text}{_SOURCES_FOOTER}" + "\n".join(lines)


def _finalize(text: str, context_chunks: list[dict] | None, plain_text: bool) -> str:
    """Normalize and attribute ``text``, but only when the caller opts in.

    ``generate_answer`` doubles as the router, the question decomposer and the
    sufficiency grader, and those callers depend on parsing the raw string
    (``route_node`` matches a single token, ``decompose_node`` splits on
    newlines, ``_ask_yes_no`` checks for a leading ``YES``). Stripping Markdown
    from their output would couple them to presentation changes, so the
    transformation is opt-in and only the answer nodes ask for it.

    Args:
        text: The raw model answer.
        context_chunks: The chunks rendered into the prompt, used to name the
            cited sources.
        plain_text: Whether to normalize the answer and append the footer.

    Returns:
        The plain-text answer with a source footer when ``plain_text`` is set,
        otherwise ``text`` exactly as the model produced it.
    """
    if not plain_text:
        return text
    return _append_sources(_normalize_output(text), context_chunks)


def _format_context(context_chunks: list[dict] | None) -> str:
    """Render retrieved chunks as ``[SOURCE n]`` blocks for the RAG prompt.

    Args:
        context_chunks: Retrieved chunks to render. Each may be a dict with
            ``text``/``source`` keys or a raw string (treated as text with no
            source). ``None`` or an empty list renders to an empty string.

    Returns:
        A newline-joined string of ``[SOURCE n]``-prefixed blocks, empty when
            no chunks are provided.
    """
    if not context_chunks:
        return ""
    blocks = []
    for index, chunk in enumerate(context_chunks, start=1):
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            source = chunk.get("source", "")
        else:
            text, source = str(chunk), ""
        source_line = f"\nSource: {display_source_name(source)}" if source else ""
        blocks.append(f"[SOURCE {index}] {text}{source_line}")
    return "\n\n".join(blocks)


def _invoke_with_retry(chain, inputs: dict) -> str:
    """Invoke ``chain`` with ``inputs``, backing off on Groq rate limits.

    Groq's on-demand tier throttles by tokens per minute; concurrent eval
    workers routinely burst past it. Retry with backoff, honoring the
    retry_after hint Groq reports in the 429 message, instead of crashing.

    Args:
        chain: The LCEL chain to invoke.
        inputs: The chain's input dict.

    Returns:
        The model's string answer.

    Raises:
        RateLimitError: If all ``MAX_RATE_LIMIT_RETRIES`` attempts are
            exhausted without a successful response.
    """
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return chain.invoke(inputs)
        except RateLimitError as exc:
            if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                raise
            wait = _retry_after_seconds(exc) or (2.0 * (attempt + 1))
            time.sleep(wait)


def generate_answer(
    question: str,
    context_chunks: list[dict] | None,
    *,
    plain_text: bool = False,
) -> str:
    """Generate an answer to ``question`` using the composed RAG chain.

    Args:
        question: The user question to answer.
        context_chunks: Retrieved chunks rendered into the prompt's context
            block. ``None`` renders an empty context.
        plain_text: Set by the answer nodes to normalize the answer to plain
            text and append a footer naming the cited sources. Left off by the
            router, decomposer and grader callers, which parse the raw output.

    Returns:
        The model's string answer.

    Raises:
        RateLimitError: If all ``MAX_RATE_LIMIT_RETRIES`` attempts are
            exhausted without a successful response.
    """
    chain = build_rag_chain()
    inputs = {"context": _format_context(context_chunks), "question": question}
    raw = _invoke_with_retry(chain, inputs)
    return _finalize(raw, context_chunks, plain_text)


def generate_partial_answer(
    question: str,
    context_chunks: list[dict] | None,
    gaps: list[str] | None = None,
    *,
    plain_text: bool = False,
) -> str:
    """Generate a best-effort answer that explicitly notes context gaps.

    Used when grading judged the retrieved context insufficient: the model
    answers only from what was retrieved and flags what it could not answer,
    instead of the graph returning a canned "don't know" string. ``gaps``
    lists the sub-questions the grader found uncovered so the answer names
    them instead of guessing.

    Args:
        question: The user question to answer.
        context_chunks: Retrieved chunks rendered into the prompt's context
            block. ``None`` renders an empty context.
        gaps: Sub-questions the retrieved context did not cover.
        plain_text: Set by the answer nodes to normalize the answer to plain
            text and append a footer naming the cited sources.

    Returns:
        The model's string answer.

    Raises:
        RateLimitError: If all ``MAX_RATE_LIMIT_RETRIES`` attempts are
            exhausted without a successful response.
    """
    chain = build_partial_answer_chain()
    note = ""
    if gaps:
        joined = "\n".join(f"- {gap}" for gap in gaps)
        note = (
            "The retrieved context does NOT cover the following parts. You MUST "
            "explicitly state you could not answer them and must not guess:\n"
            + joined
        )
    inputs = {
        "context": _format_context(context_chunks),
        "question": question,
        "note": note,
    }
    raw = _invoke_with_retry(chain, inputs)
    return _finalize(raw, context_chunks, plain_text)


def _retry_after_seconds(exc: RateLimitError) -> float | None:
    """Extract the ``retry in Xs`` hint from a Groq 429 message, if present.

    Args:
        exc: The raised rate-limit error whose body may carry the hint.

    Returns:
        The suggested wait time in seconds that Groq reported, or ``None``
            when the message carries no parseable hint.
    """
    match = _RETRY_AFTER_RE.search(str(getattr(exc, "body", "") or exc))
    if match:
        return float(match.group(1))
    return None
