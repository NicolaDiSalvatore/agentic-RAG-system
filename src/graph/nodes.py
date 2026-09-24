"""
Individual node functions for the LangGraph agentic RAG graph.
"""


from .state import GraphState
from src.llm.langchain_llm import generate_answer, generate_partial_answer
from src.indexing.retriever import build_hybrid_retriever, build_reranker
from src.indexing.build_index import build_index, load_corpus_nodes
from src.config import settings
from pathlib import Path
import math
import threading

DATA_PATH = Path(__file__).parents[2] / "data" / "gutenqa_chunks.parquet"

MAX_RETRIES = settings.max_retries

_index = None
_retriever = None
_reranker = None
_retriever_lock = threading.Lock()


def _get_retriever():
    """Build the hybrid retriever once per process and cache it.

    Thread-safe: the eval harness invokes the graph from a thread pool, and a
    naive ``is None`` check lets multiple workers race past it and each open
    their own Qdrant client on the same local storage folder. Qdrant local
    mode takes an exclusive file lock, so the extra clients all fail with
    ``AlreadyLocked``. The lock serializes construction so only one client /
    index / retriever is ever built.

    The reranker is built alongside and cached identically. Cross-encoder
    models are big and are downloaded lazily on first use, so every process
    pays the load cost once; if the model cannot be loaded (no cache, no
    network) retrieval degrades gracefully to the fused ranking.
    """
    global _index, _retriever, _reranker
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _index = build_index(str(DATA_PATH))
                # Qdrant stores node text in its payloads and keeps the index
                # docstore empty, so fall back to payload reconstruction for BM25.
                _nodes = list(_index.docstore.docs.values()) or load_corpus_nodes(_index)
                _retriever = build_hybrid_retriever(_index, _nodes)
                try:
                    _reranker = build_reranker()
                except Exception:
                    _reranker = None
    return _retriever


def _normalize_whitespace(text: str) -> str:
    """Collapse Unicode spaces (narrow no-break space, etc.) into plain spaces.

    The decomposer LLM occasionally emits words separated by U+202F; those
    cleave query tokens apart for both BM25 and the dense encoder, so retrieval
    and grading see ``Moby\\u202fDick`` instead of ``Moby Dick``.
    """
    return " ".join(text.replace("\u202f", " ").replace("\u00a0", " ").split())


def route_node(state: GraphState) -> dict:
    question = state["question"]

    prompt = f"""
    Classify the following question into exactly one category:

    - no_retrieval: can be answered without retrieving documents
    - simple: requires retrieval but is a straightforward question
    - complex: requires retrieval and involves multiple steps or concepts

    Question: {question}

    Return only one of:
    no_retrieval
    simple
    complex
    """

    route = generate_answer(prompt, None).strip().lower()

    if route not in {"no_retrieval", "simple", "complex"}:
        route = "simple"

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

    response = generate_answer(prompt, None)

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

    return {"sub_questions": [_normalize_whitespace(sq) for sq in sub_questions]}


def _rerank(hits, query: str):
    """Order fused hits with the cross-encoder reranker when available."""
    if not hits or _reranker is None:
        return hits
    return _reranker.postprocess_nodes(hits, query_str=query)


def retrieve_node(state: GraphState) -> dict:
    retriever = _get_retriever()

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
    for query in queries:
        hits = list(retriever.retrieve(query))
        pools.append((query, _rerank(hits, query)))

    seen = set()
    chunks = []

    def take(node) -> dict | None:
        text = node.get_content().strip()
        if not text or text in seen:
            return None
        seen.add(text)
        return {
            "text": text,
            "source": node.metadata.get("file_name", node.metadata.get("file_path", "")),
            "query": None,
        }

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

    return {"retrieved_chunks": chunks}


SUFFICIENCY_PROMPT = """
Decide whether the provided context is sufficient to fully answer the question.

Context: {context}

Question: {question}

Answer with exactly one word: YES or NO
"""

SUB_QUESTION_PROMPT = """
Decide whether the provided context contains enough relevant information to
answer the sub-question. The full question is broader, so relevant coverage of
this sub-question is enough; the context does not need to answer everything.

Context: {context}

Sub-question: {sub_question}

Answer with exactly one word: YES or NO
"""


def _ask_yes_no(prompt: str) -> bool:
    return generate_answer(prompt, None).strip().upper().startswith("YES")


def grade_node(state: GraphState) -> dict:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])

    context = "\n\n".join(
        f"SOURCE {i}:{chunk['text']}" for i, chunk in enumerate(chunks, start=1)
    )
    if not context.strip():
        return {
            "context_sufficient": False,
            "retry_count": state.get("retry_count", 0) + 1,
            "grade_details": [{"sub_question": sq, "covered": False} for sq in state.get("sub_questions") or [question]],
        }

    if state.get("route") == "complex":
        # Grade each sub-question separately against the fused cross-book pile.
        # A multi-part comparison is only "sufficient" when EVERY sub-question
        # is covered: a missing half (e.g. Nemo's motives) must force a retry /
        # partial answer instead of being masked by majority-vote on the parts
        # that happen to be well covered.
        sub_questions = state.get("sub_questions") or [question]
        grade_details = [
            {
                "sub_question": sq,
                "covered": _ask_yes_no(
                    SUB_QUESTION_PROMPT.format(context=context, sub_question=sq)
                ),
            }
            for sq in sub_questions
        ]
        context_sufficient = all(item["covered"] for item in grade_details)
    else:
        context_sufficient = _ask_yes_no(
            SUFFICIENCY_PROMPT.format(context=context, question=question)
        )
        grade_details = [{"sub_question": question, "covered": context_sufficient}]

    return {
        "context_sufficient": context_sufficient,
        "retry_count": state.get("retry_count", 0) + 1,
        "grade_details": grade_details,
    }


def generate_node(state: GraphState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if state.get("context_sufficient"):
        answer = generate_answer(state["question"], chunks)
    elif chunks:
        # Best-effort partial answer that explicitly notes gaps instead of
        # returning a canned "don't know" string. Grade details pinpoint which
        # sub-questions the context failed to cover so the answer names them.
        uncovered = [
            item["sub_question"]
            for item in (state.get("grade_details") or [])
            if not item.get("covered")
        ]
        answer = generate_partial_answer(state["question"], chunks, gaps=uncovered)
    else:
        answer = "I don't have enough context to answer that question."

    return {"answer": answer}


def no_retrieval_node(state: GraphState) -> dict:
    return {"answer": generate_answer(state["question"], None)}
