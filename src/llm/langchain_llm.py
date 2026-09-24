"""
Wrap the hosted LLM using LangChain's ChatGroq + LCEL prompt composition.
"""

import re
import ssl
import time
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

RAG_SYSTEM_PROMPT = """You are a helpful assistant.
- When a context block is provided, answer using ONLY that context.
- Cite the relevant sources as [SOURCE n] for every factual claim you make.
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
        source_line = f"\nSource: {source}" if source else ""
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


def generate_answer(question: str, context_chunks: list[dict] | None) -> str:
    """Generate an answer to ``question`` using the composed RAG chain.

    Args:
        question: The user question to answer.
        context_chunks: Retrieved chunks rendered into the prompt's context
            block. ``None`` renders an empty context.

    Returns:
        The model's string answer.

    Raises:
        RateLimitError: If all ``MAX_RATE_LIMIT_RETRIES`` attempts are
            exhausted without a successful response.
    """
    chain = build_rag_chain()
    inputs = {"context": _format_context(context_chunks), "question": question}
    return _invoke_with_retry(chain, inputs)


def generate_partial_answer(
    question: str,
    context_chunks: list[dict] | None,
    gaps: list[str] | None = None,
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
    return _invoke_with_retry(chain, inputs)


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
