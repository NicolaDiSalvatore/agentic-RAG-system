"""Wrap configured LLM providers using LangChain + LCEL prompt composition."""

import logging
import random
import re
import ssl
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import httpx
import truststore
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src import quota
from src.config import settings
from src.observability import metrics, tracing
from src.resilience import (
    PERMANENT,
    RATE_LIMITED,
    TIMEOUT,
    UNKNOWN,
    LLMUnavailable,
    RequestContext,
    breaker_for,
    classify_error,
    current_context,
)
from src.schemas import Citation

logger = logging.getLogger("rag.llm")

# Offline tools have no deadline, so they wait out rate limits instead of
# falling back; a request-scoped call only waits this long before moving on.
MAX_RATE_LIMIT_RETRIES = 6
MAX_REQUEST_RATE_LIMIT_WAIT_SECONDS = 2.0
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


@dataclass(frozen=True)
class LLMTarget:
    """One provider and model the service can call."""

    provider: str
    model: str


def _primary_target() -> LLMTarget:
    model = {
        "groq": settings.groq_model,
        "openai": settings.openai_model,
        "ollama": settings.ollama_model,
    }[settings.llm_provider]
    return LLMTarget(settings.llm_provider, model)


def llm_targets() -> list[LLMTarget]:
    """Return the configured provider, then each usable fallback, in order.

    A fallback whose provider has no credentials is skipped rather than being
    tried and failing on every request.

    Returns:
        The distinct targets to try, always starting with ``LLM_PROVIDER``'s.
    """
    targets = [_primary_target()]
    for entry in settings.llm_fallback_chain.split(","):
        provider, separator, model = entry.strip().partition(":")
        if not separator or not model:
            continue
        target = LLMTarget(provider.strip().lower(), model.strip())
        if target.provider not in {"groq", "openai", "ollama"}:
            logger.warning("Ignoring unknown fallback provider %r.", target.provider)
            continue
        if target.provider == "openai" and not settings.openai_api_key:
            continue
        if target not in targets:
            targets.append(target)
    return targets


@lru_cache(maxsize=8)
def _get_chat_model(provider: str, model: str) -> ChatGroq | ChatOpenAI:
    """Build a configured chat client for one target once and reuse it.

    Windows enterprise proxies often use a root CA that is trusted by the
    operating system but not by certifi, which httpx uses by default, so the
    client is wired to a truststore-backed SSL context. Retries are disabled in
    the SDKs because ``_invoke_with_fallback`` owns the retry budget.

    Args:
        provider: The provider adapter to use.
        model: The model identifier to target.

    Returns:
        A cached provider client bound to ``model``.
    """
    timeout = httpx.Timeout(
        settings.llm_attempt_timeout_seconds, connect=settings.llm_connect_timeout_seconds
    )
    http_client = httpx.Client(
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), timeout=timeout
    )
    if provider in {"openai", "ollama"}:
        is_ollama = provider == "ollama"
        return ChatOpenAI(
            model=model,
            api_key=("ollama" if is_ollama else settings.openai_api_key),
            base_url=(settings.ollama_base_url if is_ollama else settings.openai_base_url) or None,
            temperature=settings.groq_temperature,
            timeout=timeout,
            max_retries=0,
            max_tokens=settings.max_output_tokens,
            http_client=http_client,
        )
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key or None,
        temperature=settings.groq_temperature,
        model_kwargs={"seed": settings.groq_seed},
        timeout=timeout,
        max_retries=0,
        max_tokens=settings.max_output_tokens,
        http_client=http_client,
    )


def get_chat_model() -> ChatGroq | ChatOpenAI:
    """Return the client for the configured primary provider and model."""
    target = _primary_target()
    return _get_chat_model(target.provider, target.model)


@lru_cache(maxsize=1)
def get_judge_chat_model() -> ChatGroq | ChatOpenAI:
    """Build the RAGAS judge client once per process and reuse it.

    Returns:
        A cached client using the configured provider and judge model.
    """
    # The judge uses the selected adapter as well; configure a dedicated Groq
    # judge only when Groq is the active provider.
    if settings.llm_provider == "groq":
        return _get_chat_model("groq", settings.ragas_judge_model)
    return get_chat_model()


_RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

_PARTIAL_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", PARTIAL_ANSWER_SYSTEM_PROMPT.format(extra_instructions="")),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{note}"),
])


class _PromptedModel:
    """Fill a prompt, then call the model.

    An LCEL ``template | model`` chain would add a chain entry and a prompt entry
    to every trace. Calling the model directly leaves one entry per LLM attempt,
    which still shows the final messages.
    """

    def __init__(self, template: ChatPromptTemplate, model: Any) -> None:
        self._template = template
        self._model = model

    def invoke(self, inputs: dict, config: dict | None = None) -> Any:
        return self._model.invoke(self._template.format_messages(**inputs), config=config)


def build_rag_chain(target: LLMTarget, timeout: float) -> _PromptedModel:
    """Compose the RAG prompt with one target, limited to ``timeout`` seconds.

    Composition is cheap and the timeout differs per attempt, because each call
    only gets what is left of the request, so this chain is not cached.

    Args:
        target: The provider and model to call.
        timeout: The longest this attempt may take, in seconds.

    Returns:
        An object whose ``invoke`` fills the prompt and calls the model.
    """
    return _PromptedModel(
        _RAG_TEMPLATE, _get_chat_model(target.provider, target.model).bind(timeout=timeout)
    )


def build_partial_answer_chain(target: LLMTarget, timeout: float) -> _PromptedModel:
    """Compose the partial-answer prompt with one target.

    The chain answers from whatever context is available while explicitly
    noting gaps, used by ``generate_partial_answer`` when the retrieved context
    looked insufficient.

    Args:
        target: The provider and model to call.
        timeout: The longest this attempt may take, in seconds.

    Returns:
        An object whose ``invoke`` fills the prompt and calls the model.
    """
    return _PromptedModel(
        _PARTIAL_TEMPLATE, _get_chat_model(target.provider, target.model).bind(timeout=timeout)
    )


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


def validated_citations(text: str, context_chunks: list[dict] | None) -> list[dict]:
    """Return only citations that resolve to retrieved, canonical chunk IDs.

    The model controls only a numeric marker. This function resolves that
    marker against the exact retrieval result, discarding out-of-range or
    non-canonical references before they can reach an API client.
    """
    citations: list[dict] = []
    for index in cited_source_indices(split_sources_footer(text), context_chunks):
        chunk = (context_chunks or [])[index - 1]
        document_id = chunk.get("document_id")
        chunk_id = chunk.get("chunk_id")
        source_uri = chunk.get("source_uri") or chunk.get("source")
        if document_id and chunk_id and source_uri:
            citations.append(
                Citation(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    source_uri=source_uri,
                ).model_dump()
            )
    return citations


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a about after all also an and are as at be been but by can could did do does for from
    had has have he her him his how i if in into is it its may more most no not of on or our
    she should so some such than that the their them then there these they this those to
    up was we were what when where which who whom why will with would you your""".split()
)


def _content_words(text: str) -> set[str]:
    """Return a text's lower-cased words, minus stopwords and crude plurals."""
    words = _WORD_RE.findall(unicodedata.normalize("NFKC", text).lower().replace("'s", ""))
    return {
        word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in words
        if word not in _STOPWORDS
    }


@dataclass(frozen=True)
class CitationReport:
    """How well an answer's citations hold up against the retrieved chunks.

    ``validated_citations`` silently discards bad references, so its result can
    never show a failure. This report counts them, which is what a metric or a
    release gate needs. Support is a vocabulary check, not an entailment check:
    it catches citations pointing at the wrong chunk, not subtle misreadings.
    """

    citations: list[dict]
    markers_total: int
    invalid_markers: int
    non_canonical_markers: int
    cited_sentences: int
    unsupported_sentences: int
    uncited_sentences: int

    @property
    def grounded(self) -> bool:
        """Whether the answer has citations and none of them failed a check."""
        return bool(
            self.citations
            and not self.invalid_markers
            and not self.non_canonical_markers
            and not self.unsupported_sentences
        )

    @property
    def validity(self) -> float | None:
        """Share of emitted markers that resolved, or ``None`` with no markers."""
        if not self.markers_total:
            return None
        failed = self.invalid_markers + self.non_canonical_markers
        return (self.markers_total - failed) / self.markers_total


def _split_sentences(body: str) -> list[str]:
    """Split text into sentences, keeping a trailing marker with its sentence."""
    sentences: list[str] = []
    for piece in _SENTENCE_RE.split(body):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and not _CITATION_RE.sub("", piece).strip(" .,;:"):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def citation_report(text: str, context_chunks: list[dict] | None) -> CitationReport:
    """Check every citation marker in an answer against the retrieved chunks.

    Args:
        text: A finalized answer, with or without its sources footer.
        context_chunks: The chunks the answer was generated from.

    Returns:
        The counts for this answer, plus its resolved citations.
    """
    chunks = context_chunks or []
    body = split_sources_footer(text)
    markers = [int(match.group(1)) for match in _CITATION_RE.finditer(body)]
    invalid = sum(1 for index in markers if not 1 <= index <= len(chunks))

    def canonical(index: int) -> bool:
        chunk = chunks[index - 1]
        return bool(
            chunk.get("document_id")
            and chunk.get("chunk_id")
            and (chunk.get("source_uri") or chunk.get("source"))
        )

    non_canonical = sum(
        1 for index in markers if 1 <= index <= len(chunks) and not canonical(index)
    )

    cited_sentences = unsupported = uncited = 0
    for sentence in _split_sentences(body):
        indices = {int(match.group(1)) for match in _CITATION_RE.finditer(sentence)}
        words = _content_words(_CITATION_RE.sub(" ", sentence))
        if not indices:
            uncited += 1 if words else 0
            continue
        cited_sentences += 1
        valid = [index for index in indices if 1 <= index <= len(chunks)]
        if not words or not valid:
            continue
        evidence: set[str] = set()
        for index in valid:
            evidence |= _content_words(chunks[index - 1].get("text", ""))
        if len(words & evidence) / len(words) < settings.citation_min_support:
            unsupported += 1

    return CitationReport(
        citations=validated_citations(text, chunks),
        markers_total=len(markers),
        invalid_markers=invalid,
        non_canonical_markers=non_canonical,
        cited_sentences=cited_sentences,
        unsupported_sentences=unsupported,
        uncited_sentences=uncited,
    )


def strip_invalid_markers(text: str, context_chunks: list[dict] | None) -> str:
    """Remove citation markers that point at no retrieved chunk.

    A reader cannot tell a fabricated ``[SOURCE 9]`` from a real one, so it must
    not be displayed even though the report has already counted it.

    Args:
        text: A finalized answer.
        context_chunks: The chunks the answer was generated from.

    Returns:
        The answer with out-of-range markers and the space before them removed.
    """
    limit = len(context_chunks or [])

    def keep_valid(match: re.Match) -> str:
        return match.group(0) if 1 <= int(match.group(2)) <= limit else ""

    pattern = re.compile(r"(\s*)\[?\s*SOURCE\s*#?\s*(\d+)\s*\]?", re.IGNORECASE)
    return pattern.sub(keep_valid, text)


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

    ``generate_answer`` doubles as the question decomposer, which parses the raw
    string (``decompose_node`` splits on newlines). Stripping Markdown from its
    output would couple it to presentation changes, so the transformation is
    opt-in and only the answer nodes ask for it.

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


def _message_text(result: Any) -> str:
    """Return the text of a chain result, which is a chat message or a string."""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", "")
    return content if isinstance(content, str) else str(content)


def _record_success(
    target: LLMTarget, role: str, result: Any, started: float, context: RequestContext | None
) -> None:
    """Record a successful call's latency and token use everywhere it is tracked."""
    usage = getattr(result, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    labels = {"provider": target.provider, "model": target.model}
    metrics.LLM_CALLS.labels(**labels, role=role, status="ok").inc()
    metrics.LLM_SECONDS.labels(**labels).observe(time.monotonic() - started)
    metrics.LLM_TOKENS.labels(**labels, direction="input").inc(input_tokens)
    metrics.LLM_TOKENS.labels(**labels, direction="output").inc(output_tokens)
    quota.record(input_tokens + output_tokens)
    spent = quota.usage()
    metrics.QUOTA_USED.labels(unit="tokens").set(spent["tokens"])
    metrics.QUOTA_USED.labels(unit="requests").set(spent["requests"])
    if context is not None:
        context.input_tokens += input_tokens
        context.output_tokens += output_tokens
        context.models_used.append(f"{target.provider}:{target.model}")


def _attempt_timeout(context: RequestContext | None) -> float:
    """Return this attempt's time limit, raising when too little time is left."""
    remaining = context.remaining() if context is not None else None
    if remaining is None:
        return settings.llm_attempt_timeout_seconds
    if remaining < settings.llm_min_call_seconds:
        raise LLMUnavailable("deadline")
    return min(settings.llm_attempt_timeout_seconds, remaining)


def _invoke_with_fallback(
    build_chain: Callable[[LLMTarget, float], Any], inputs: dict, role: str
) -> str:
    """Call providers in order until one answers or the request's limits run out.

    Timeouts move straight to the next target, because waiting for one provider
    twice would spend the whole request and leave nothing for the fallback.
    Other transient errors get a bounded retry with backoff and jitter. Each
    attempt counts against the request's call budget and the daily quota.

    Args:
        build_chain: Builds the chain for a target and an attempt timeout.
        inputs: The chain's input dict.
        role: What the call is for, used as a metrics label.

    Returns:
        The model's string answer.

    Raises:
        LLMUnavailable: When no provider could answer within the limits.
    """
    context = current_context()
    max_attempts = (
        MAX_RATE_LIMIT_RETRIES if context is None else settings.llm_max_attempts_per_provider
    )
    config = {"callbacks": tracing.callbacks(), "run_name": role}
    for target in llm_targets():
        labels = {"provider": target.provider, "model": target.model}
        breaker = breaker_for(
            target.provider,
            target.model,
            settings.breaker_failure_threshold,
            settings.breaker_reset_seconds,
        )
        for attempt in range(max_attempts):
            if not breaker.allow():
                metrics.LLM_CALLS.labels(**labels, role=role, status="breaker_open").inc()
                break
            if context is not None and context.budget_exhausted():
                raise LLMUnavailable("call_budget")
            if quota.exhausted():
                metrics.QUOTA_EXHAUSTED.inc()
                raise LLMUnavailable("daily_quota")
            timeout = _attempt_timeout(context)
            if context is not None:
                context.llm_calls += 1
            started = time.monotonic()
            try:
                result = build_chain(target, timeout).invoke(inputs, config=config)
            except Exception as exc:
                kind = classify_error(exc)
                if kind == UNKNOWN:
                    raise
                metrics.LLM_CALLS.labels(**labels, role=role, status=kind).inc()
                logger.warning(
                    "LLM call failed: %s %s role=%s kind=%s error=%s",
                    target.provider, target.model, role, kind, type(exc).__name__,
                )
                wait = None if attempt == max_attempts - 1 else _retry_delay(
                    kind, exc, attempt, context
                )
                if wait is None:
                    if kind == RATE_LIMITED:
                        hint = _retry_after_seconds(exc)
                        breaker.open_for(hint if hint else settings.breaker_reset_seconds)
                    else:
                        breaker.record_failure()
                    break
                breaker.record_failure()
                time.sleep(wait)
            else:
                breaker.record_success()
                _record_success(target, role, result, started, context)
                return _message_text(result)
            finally:
                metrics.BREAKER_STATE.labels(**labels).set(
                    0 if breaker.state == breaker.CLOSED else 1
                )
    raise LLMUnavailable("all_providers_failed")


def _retry_delay(
    kind: str, exc: Exception, attempt: int, context: RequestContext | None
) -> float | None:
    """Return how long to wait before retrying a target, or ``None`` to move on."""
    if kind in {TIMEOUT, PERMANENT}:
        return None
    hint = _retry_after_seconds(exc)
    delay = hint if hint is not None else (2.0 * (attempt + 1)) * random.uniform(0.5, 1.0)
    if context is None:
        return delay
    if kind == RATE_LIMITED and delay > MAX_REQUEST_RATE_LIMIT_WAIT_SECONDS:
        return None
    delay = min(delay, MAX_REQUEST_RATE_LIMIT_WAIT_SECONDS)
    remaining = context.remaining()
    if remaining is not None and remaining - delay < settings.llm_min_call_seconds:
        return None
    return delay


def generate_answer(
    question: str,
    context_chunks: list[dict] | None,
    *,
    plain_text: bool = False,
    role: str = "generate",
) -> str:
    """Generate an answer to ``question`` using the composed RAG chain.

    Args:
        question: The user question to answer.
        context_chunks: Retrieved chunks rendered into the prompt's context
            block. ``None`` renders an empty context.
        plain_text: Set by the answer nodes to normalize the answer to plain
            text and append a footer naming the cited sources. Left off by the
            decomposer, which parses the raw output.
        role: What the call is for, used as a metrics label.

    Returns:
        The model's string answer.

    Raises:
        LLMUnavailable: If no provider could answer within the request's limits.
    """
    inputs = {"context": _format_context(context_chunks), "question": question}
    raw = _invoke_with_fallback(lambda t, to: build_rag_chain(t, to), inputs, role)
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
        LLMUnavailable: If no provider could answer within the request's limits.
    """
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
    raw = _invoke_with_fallback(lambda t, to: build_partial_answer_chain(t, to), inputs, "partial")
    return _finalize(raw, context_chunks, plain_text)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract the ``retry in Xs`` hint from a Groq 429 message, if present.

    Args:
        exc: The raised rate-limit exception whose body or headers may carry a hint.

    Returns:
        The suggested wait time in seconds that Groq reported, or ``None``
            when the message carries no parseable hint.
    """
    response = getattr(exc, "response", None)
    retry_after = getattr(response, "headers", {}).get("retry-after") if response else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    match = _RETRY_AFTER_RE.search(str(getattr(exc, "body", "") or exc))
    if match:
        return float(match.group(1))
    return None
