"""
Individual node functions for the LangGraph agentic RAG graph.
"""



from .fallbacks import extractive_answer
from .state import GraphState
from src.llm.langchain_llm import (
    citation_report,
    generate_answer,
    generate_partial_answer,
    strip_invalid_markers,
)
from src.indexing.retriever import build_hybrid_retriever, build_reranker
from src.indexing.build_index import load_existing_index, load_corpus_nodes
from src.config import settings
from src.observability import metrics
from src.resilience import LLMUnavailable
from pathlib import Path
import logging
import math
import re
import threading

logger = logging.getLogger("rag.graph")

DATA_PATH = Path(__file__).parents[2] / "data" / "gutenqa_chunks.parquet"

MAX_RETRIES = settings.max_retries

_index = None
_retrievers: dict[tuple[str, tuple[str, ...]], object] = {}
_reranker = None
_retriever_lock = threading.Lock()


class _EmptyRetriever:
    """Retriever used when an identity has no permitted chunks."""

    def retrieve(self, query: str) -> list:
        return []


def _permitted(metadata: dict, tenant_id: str, roles: set[str]) -> bool:
    """Return whether a chunk is visible to an identity.

    Missing governance metadata fails closed: a legacy chunk with no
    ``tenant_id`` or ``allowed_roles`` is never served.
    """
    if metadata.get("tenant_id") != tenant_id:
        return False
    raw_roles = metadata.get("allowed_roles") or []
    allowed = {raw_roles} if isinstance(raw_roles, str) else set(raw_roles)
    return bool(allowed.intersection(roles))


def _get_reranker():
    """Build the reranker once when the profile enables it."""
    global _reranker
    if _reranker is None and settings.enable_reranker:
        try:
            _reranker = build_reranker()
        except Exception:
            _reranker = None
    return _reranker


def _get_retriever(tenant_id: str, roles: list[str]):
    """Build and cache a retriever scoped to one tenant/role identity.

    The dense retriever is filtered in Qdrant and the BM25 corpus is built from
    only the visible nodes, so disallowed text is never retrieved or embedded
    in the prompt. Retrievers are cached per identity because BM25 construction
    over the corpus is expensive; the number of distinct identities is small
    in this single-tenant deployment.

    Args:
        tenant_id: The caller's tenant.
        roles: The caller's roles.

    Returns:
        A retriever that only returns chunks visible to the identity.
    """
    global _index
    role_key = tuple(sorted({role for role in roles if role}))
    key = (tenant_id, role_key)
    retriever = _retrievers.get(key)
    if retriever is not None:
        return retriever
    with _retriever_lock:
        retriever = _retrievers.get(key)
        if retriever is not None:
            return retriever
        if _index is None:
            _index = load_existing_index()
        # Qdrant stores node text in its payloads and keeps the index
        # docstore empty, so fall back to payload reconstruction for BM25.
        nodes = list(_index.docstore.docs.values()) or load_corpus_nodes(_index)
        visible = [
            node
            for node in nodes
            if _permitted(node.metadata or {}, tenant_id, set(role_key))
        ]
        if not visible:
            retriever = _EmptyRetriever()
        else:
            retriever = build_hybrid_retriever(
                _index, visible, tenant_id=tenant_id, roles=list(role_key)
            )
        _retrievers[key] = retriever
        _get_reranker()
    return retriever


def _normalize_whitespace(text: str) -> str:
    """Collapse Unicode spaces (narrow no-break space, etc.) into plain spaces.

    The decomposer LLM occasionally emits words separated by U+202F; those
    cleave query tokens apart for both BM25 and the dense encoder, so retrieval
    and grading see ``Moby\\u202fDick`` instead of ``Moby Dick``.
    """
    return " ".join(text.replace("\u202f", " ").replace("\u00a0", " ").split())


# Deliberately narrow. Words like "and" or "why" are not signals in a corpus of
# titles such as "Pride and Prejudice"; a question these rules miss is escalated
# later if its retrieved evidence turns out weak.
_COMPLEX_RE = re.compile(
    r"\b(compare[ds]?|comparison|contrast(ed|s)?|differ(s|ed|ence|ences)?|similarit(y|ies)"
    r"|versus|vs\.?)\b",
    re.IGNORECASE,
)


def route_node(state: GraphState) -> dict:
    """Route a question from its wording alone, with no LLM call.

    Every question is answered from documents. ``complex`` is reserved for
    questions that clearly have several parts; everything else starts as
    ``simple`` and relies on the evidence check to escalate.
    """
    question = state["question"]
    is_complex = _COMPLEX_RE.search(question) or question.count("?") > 1
    route = "complex" if is_complex else "simple"
    metrics.ROUTES.labels(route=route).inc()
    return {"route": route}


def decompose_node(state: GraphState) -> dict:
    question = state["question"]

    prompt = f"""Break the user's question into smaller, independent sub-questions
    that are necessary to answer the original question.

    Rules:
    - Return ONLY the sub-questions, one per line.
    - Each sub-question must be clear, self-contained, and a real question
      ending with a question mark.
    - Do not answer the sub-questions.
    - Do not create unnecessary sub-questions.
    - If the question does not need decomposition, return the original question.
    - Never produce statements like "I don't know" or comments about the
      context; only emit interrogative sentences.

    Question: {question}
    """

    try:
        response = generate_answer(prompt, None, role="decompose")
    except LLMUnavailable as exc:
        # Decomposition is an optimization: the original question still works.
        logger.warning("Decomposition skipped: %s", exc.reason)
        return {"sub_questions": [_normalize_whitespace(question)]}

    sub_questions = []
    for line in response.splitlines():
        line = line.strip("- ").strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(("i don", "i do not", "i can")) or any(
            phrase in lower for phrase in ("i don't know", "insufficient")
        ):
            continue
        if not line.endswith("?"):
            continue
        sub_questions.append(line)

    if not sub_questions:
        sub_questions = [question]

    return {
        "sub_questions": [
            _normalize_whitespace(sq)
            for sq in sub_questions[: settings.max_sub_questions]
        ]
    }


def escalate_node(state: GraphState) -> dict:
    """Decompose a ``simple`` question whose first retrieval found weak evidence."""
    metrics.ESCALATIONS.inc()
    return {**decompose_node(state), "route": "complex", "escalated": True}


def _rerank(hits, query: str):
    """Order fused hits with the cross-encoder reranker when available."""
    if not hits or _reranker is None:
        return hits
    return _reranker.postprocess_nodes(hits, query_str=query)


def retrieve_node(state: GraphState) -> dict:
    tenant_id = state.get("tenant_id") or "tenant-demo"
    roles = [role for role in (state.get("roles") or ["public"]) if role]
    retriever = _get_retriever(tenant_id, roles)

    queries = state.get("sub_questions") or [state["question"]]

    # Widen the total budget on each retry so a re-fetch pulls more context for
    # the grader to evaluate. The budget is split evenly across sub-questions
    # (with a per-query cap) so no single sub-question can monopolize it and
    # every part of a multi-part question contributes sources.
    attempt = state.get("retry_count", 0)
    budget = settings.top_k_reranked + attempt * settings.top_k_reranked
    per_query_cap = max(1, math.ceil(budget / len(queries)))

    # Fetch and rerank every sub-question independently, holding its pool back
    # so the guarantee pass below can pull at least one chunk per sub-question
    # before any query's cap fills the total budget.
    pools: list[tuple[str, list]] = []
    evidence: list[dict] = []
    for query in queries:
        hits = list(retriever.retrieve(query))
        # Read before reranking, because the reranker replaces scores.
        top_dense = getattr(hits[0], "top_dense_score", None) if hits else None
        if top_dense is not None:
            metrics.TOP_DENSE_SCORE.observe(top_dense)
        evidence.append({"query": query, "top_dense_score": top_dense})
        pools.append((query, _rerank(hits, query)))

    seen = set()
    chunks: list[dict] = []
    context_chars = 0
    role_set = set(roles)

    def take(node) -> dict | None:
        nonlocal context_chars
        metadata = node.metadata or {}
        if not _permitted(metadata, tenant_id, role_set):
            return None
        text = node.get_content().strip()
        remaining = settings.max_context_chars - context_chars
        if not text or text in seen or remaining <= 0:
            return None
        seen.add(text)
        text = text[:remaining]
        context_chars += len(text)
        chunk = {
            "text": text,
            "source": metadata.get("file_name", metadata.get("file_path", "")),
            "query": None,
        }
        # Keep legacy output compact while exposing canonical source identity
        # whenever the indexed corpus provides it.
        for key in ("document_id", "chunk_id", "source_uri", "index_version"):
            if metadata.get(key):
                chunk[key] = metadata[key]
        return chunk

    # Pass 1: guarantee one chunk per sub-question (cross-book coverage) so the
    # second half of a comparison question is never starved out by the first.
    for query, hits in pools:
        if len(chunks) >= budget:
            break
        for node_with_score in hits:
            if len(chunks) >= budget:
                break
            chunk = take(node_with_score.node)
            if chunk is not None:
                chunk["query"] = query
                chunks.append(chunk)
                break

    # Pass 2: round-robin fill the remaining budget across sub-questions so no
    # single query's cap empties the pool for later ones.
    if len(chunks) < budget:
        for query, hits in pools:
            taken = 0
            for node_with_score in hits:
                if len(chunks) >= budget:
                    break
                if taken >= per_query_cap:
                    break
                chunk = take(node_with_score.node)
                if chunk is not None:
                    taken += 1
                    chunk["query"] = query
                    chunks.append(chunk)

    return {"retrieved_chunks": chunks, "evidence": evidence}


def grade_node(state: GraphState) -> dict:
    """Judge each query's coverage from retrieval evidence, with no LLM call.

    A query is covered when its best cosine similarity reaches a threshold
    calibrated on this corpus. Fused and BM25 scores cannot make this decision:
    RRF scores depend only on ranks, and BM25 scores off-topic queries as highly
    as real ones. A missing score fails closed as not covered.
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1
    if not state.get("retrieved_chunks"):
        return {
            "context_sufficient": False,
            "retry_count": retry_count,
            "grade_details": [
                {"sub_question": sq, "covered": False}
                for sq in state.get("sub_questions") or [question]
            ],
        }

    def covered(item: dict) -> bool:
        score = item.get("top_dense_score")
        return score is not None and score >= settings.evidence_min_dense_score

    evidence = state.get("evidence") or [{"query": question}]
    # Every sub-question must be covered: a well-covered half of a comparison
    # must not hide that the other half has no support.
    grade_details = [
        {"sub_question": item["query"], "covered": covered(item)} for item in evidence
    ]
    return {
        "context_sufficient": all(item["covered"] for item in grade_details),
        "retry_count": retry_count,
        "grade_details": grade_details,
    }


def generate_node(state: GraphState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return _finish(
            "I don't have enough context to answer that question.", [], "refused", None
        )
    try:
        if state.get("context_sufficient"):
            answer = generate_answer(state["question"], chunks, plain_text=True)
            status = "ok"
        else:
            # Best-effort partial answer that explicitly notes gaps instead of
            # returning a canned "don't know" string. Grade details pinpoint which
            # sub-questions the context failed to cover so the answer names them.
            uncovered = [
                item["sub_question"]
                for item in (state.get("grade_details") or [])
                if not item.get("covered")
            ]
            answer = generate_partial_answer(
                state["question"], chunks, gaps=uncovered, plain_text=True
            )
            status = "partial"
    except LLMUnavailable as exc:
        # The evidence is already in hand, so a provider outage or a spent quota
        # still gets a cited answer rather than an error.
        logger.warning("Serving an extractive answer: %s", exc.reason)
        metrics.FALLBACKS.labels(kind=f"extractive_{exc.reason}").inc()
        return _finish(extractive_answer(chunks), chunks, "extractive", "extractive")
    return _finish(answer, chunks, status, None)


def _finish(answer: str, chunks: list[dict], status: str, fallback: str | None) -> dict:
    """Validate an answer's citations and record its outcome."""
    report = citation_report(answer, chunks)
    # A document answer is grounded only when its citations resolve to the exact
    # evidence passed to the model and none of them failed a check.
    invalid = report.invalid_markers + report.non_canonical_markers
    metrics.CITATION_MARKERS.labels(result="invalid").inc(invalid)
    metrics.CITATION_MARKERS.labels(result="valid").inc(report.markers_total - invalid)
    metrics.UNSUPPORTED_CLAIMS.inc(report.unsupported_sentences)
    metrics.ANSWER_STATUS.labels(status=status).inc()
    metrics.GROUNDED.labels(grounded=str(report.grounded).lower()).inc()
    return {
        "answer": strip_invalid_markers(answer, chunks),
        "citations": report.citations,
        "grounded": report.grounded,
        "answer_status": status,
        "fallback": fallback,
    }
