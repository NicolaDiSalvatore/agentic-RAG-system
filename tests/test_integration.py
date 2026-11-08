"""End-to-end tests over a real index: only the embedding model and LLM are faked.

Parquet ingestion, embedded Qdrant, the manifest, the node cache, BM25, RRF, the
graph and the HTTP layer are all real. The mock-heavy unit tests stayed green
while the service could not answer at all, because nothing exercised this path.
"""

import hashlib
import math
import re
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

import src.api.main as api
from src import quota
from src.config import settings
from src.graph import nodes
from src.indexing import build_index as index_module
from src.indexing.build_index import build_index, close_index
from src.llm import langchain_llm as llm
from src.llm.langchain_llm import LLMTarget

DIMENSIONS = 384
BOOK_A = "Around_the_World_in_Eighty_Days_-_Jules_Verne"
BOOK_B = "Twenty_Thousand_Leagues_-_Jules_Verne"
ROWS = [
    (BOOK_A, "1", "Phileas Fogg lived at Number 7 Saville Row in London."),
    (BOOK_A, "2", "Fogg wagered that he could travel around the world in eighty days."),
    (BOOK_B, "1", "Captain Nemo commanded the submarine Nautilus beneath the sea."),
    (BOOK_B, "2", "Nemo built the Nautilus to escape the society he despised."),
    (BOOK_B, "3", "The harpooner Ned Land wanted to leave the submarine."),
]


class BagOfWordsEmbedding(BaseEmbedding):
    """A deterministic embedding, so shared words produce a high cosine score."""

    _dimensions: int = PrivateAttr(default=DIMENSIONS)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % DIMENSIONS] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _embed(self, texts):
        return [self._vector(text) for text in texts]

    def _get_query_embedding(self, query):
        return self._vector(query)

    def _get_text_embedding(self, text):
        return self._vector(text)

    async def _aget_query_embedding(self, query):
        return self._vector(query)


class ScriptedLLM:
    """Replaces provider calls, answering by what each prompt asks for."""

    def __init__(self, answer="Fogg lived at Number 7 Saville Row [SOURCE 1].", error=None):
        self.answer = answer
        self.error = error
        self.contexts = []
        self.decompositions = 0

    def build(self, target, timeout):
        return MagicMock(invoke=self._invoke)

    def _invoke(self, inputs, config=None):
        if self.error is not None:
            raise self.error
        if "Break the user's question" in inputs["question"]:
            self.decompositions += 1
            return "Where did Phileas Fogg live?\nWho built the Nautilus?"
        self.contexts.append(inputs["context"])
        return self.answer


def _reset_module_caches():
    close_index()
    index_module._cached_index = None
    index_module._cached_nodes = None
    index_module._manifest_cache = None
    index_module._embed_cache = None
    nodes._index = None
    nodes._retrievers.clear()


def _write_corpus(path, rows):
    pd.DataFrame(rows, columns=["Book Name", "Chapter", "Chunk"]).to_parquet(path)
    return str(path)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolate all on-disk index state and use the deterministic embedding."""
    _reset_module_caches()
    cache = tmp_path / ".cache"
    monkeypatch.setattr(index_module, "QDRANT_PATH", cache / "qdrant")
    monkeypatch.setattr(index_module, "NODE_CACHE_ROOT", cache / "corpus_nodes")
    monkeypatch.setattr(index_module, "EMBED_CACHE_ROOT", cache / "corpus_embeddings")
    monkeypatch.setattr(index_module, "MANIFEST_ROOT", cache / "manifests")
    monkeypatch.setattr(index_module, "get_embed_model", BagOfWordsEmbedding)
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_collection", "test_corpus")
    monkeypatch.setattr(settings, "index_version", "v1")
    monkeypatch.setattr(settings, "embedding_model", "test-embedding")
    monkeypatch.setattr(settings, "enable_reranker", False)
    monkeypatch.setattr(settings, "evidence_min_dense_score", 0.2)
    monkeypatch.setattr(llm, "llm_targets", lambda: [LLMTarget("groq", "test")])
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    yield tmp_path
    _reset_module_caches()


@pytest.fixture
def corpus(workspace):
    path = _write_corpus(workspace / "chunks.parquet", ROWS)
    build_index(path)
    return path


def _use_llm(monkeypatch, scripted):
    monkeypatch.setattr(llm, "build_rag_chain", scripted.build)
    monkeypatch.setattr(llm, "build_partial_answer_chain", scripted.build)


def _query(question):
    return TestClient(api.app).post("/query", json={"question": question})


def test_ingested_corpus_answers_with_a_resolvable_citation(corpus, monkeypatch):
    scripted = ScriptedLLM()
    _use_llm(monkeypatch, scripted)

    response = _query("Where did Phileas Fogg live?")

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "simple"
    assert body["answer_status"] == "ok"
    assert body["grounded"] is True
    assert body["fallback"] is None
    assert body["index_version"] == "v1"
    assert body["citations"][0]["source_uri"] == BOOK_A
    assert "Saville Row" in scripted.contexts[0]
    assert scripted.decompositions == 0  # The simple path spends one LLM call.
    assert len(scripted.contexts) == 1


def test_readiness_follows_ingestion(workspace, monkeypatch):
    client = TestClient(api.app)
    assert client.get("/health/ready").status_code == 503
    build_index(_write_corpus(workspace / "chunks.parquet", ROWS))
    assert client.get("/health/ready").status_code == 200


def test_fusion_returns_rank_scores_and_the_dense_signal(corpus):
    # Regression: raw-score fusion sorted BM25's scale above every cosine score.
    retriever = nodes._get_retriever("tenant-demo", ["public"])
    hits = retriever.retrieve("Who commanded the Nautilus?", ranking="rrf")
    assert hits[0].node.get_content().startswith("Captain Nemo")
    assert all(hit.score <= 2.0 / 61 for hit in hits)  # Rank scores, not BM25's.
    assert hits[0].top_dense_score is not None and hits[0].top_dense_score <= 1.0


def test_node_cache_rejects_nodes_from_an_older_ingest(workspace, monkeypatch):
    # Regression: the cache was checked by point count alone, so a corpus with
    # the same size kept serving the old text.
    source = workspace / "chunks.parquet"
    build_index(_write_corpus(source, ROWS))
    first = nodes._get_retriever("tenant-demo", ["public"])
    assert first.retrieve("Saville Row")

    replaced = [(book, chapter, text.replace("Saville Row", "Baker Street"))
                for book, chapter, text in ROWS]
    _reset_module_caches()
    _write_corpus(source, replaced)
    index_module._write_manifest(source, len(ROWS))
    _reset_module_caches()

    stale_identity = index_module._node_cache_identity(len(ROWS))
    assert index_module._load_node_cache(stale_identity) is None


def test_node_cache_is_reused_for_an_unchanged_index(corpus):
    retriever = nodes._get_retriever("tenant-demo", ["public"])
    assert retriever.retrieve("Saville Row")
    assert index_module._node_cache_path().is_file()

    identity = index_module._node_cache_identity(len(ROWS))
    cached = index_module._load_node_cache(identity)
    assert cached is not None and len(cached) == len(ROWS)
    assert all(node.metadata.get("tenant_id") for node in cached)


def test_legacy_node_cache_formats_are_ignored(corpus):
    import pickle

    path = index_module._node_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(["old", "list", "format"]))
    assert index_module._load_node_cache(index_module._node_cache_identity(3)) is None


def test_building_a_new_version_keeps_the_old_one_for_rollback(corpus, workspace, monkeypatch):
    # Regression: building a missing collection wiped all of local Qdrant.
    _reset_module_caches()
    monkeypatch.setattr(settings, "index_version", "v2")
    build_index(_write_corpus(workspace / "v2.parquet", ROWS[:2]))
    assert index_module.index_is_ready()

    _reset_module_caches()
    monkeypatch.setattr(settings, "index_version", "v1")
    assert index_module.index_is_ready()
    index = index_module.load_existing_index()
    client = index.vector_store.client
    assert client.count(collection_name="test_corpus__v1", exact=True).count == len(ROWS)
    assert client.count(collection_name="test_corpus__v2", exact=True).count == 2


def test_interrupted_ingest_is_finished_by_the_next_run(workspace):
    source = workspace / "chunks.parquet"
    build_index(_write_corpus(source, ROWS[:2]))
    _reset_module_caches()
    index = build_index(_write_corpus(source, ROWS))
    count = index.vector_store.client.count(collection_name="test_corpus__v1", exact=True).count
    assert count == len(ROWS)
    assert index_module.read_manifest()["point_count"] == len(ROWS)


def test_reingesting_unchanged_input_adds_no_points(corpus):
    _reset_module_caches()
    index = build_index(corpus)
    count = index.vector_store.client.count(collection_name="test_corpus__v1", exact=True).count
    assert count == len(ROWS)


def test_comparison_question_decomposes_up_front(corpus, monkeypatch):
    scripted = ScriptedLLM(answer="Fogg lived in London [SOURCE 1]. Nemo built it [SOURCE 2].")
    _use_llm(monkeypatch, scripted)

    body = _query("Compare where Phileas Fogg lived and who built the Nautilus").json()

    assert body["route"] == "complex"
    assert scripted.decompositions == 1
    assert "Saville Row" in scripted.contexts[0]
    assert "Nautilus" in scripted.contexts[0]


def test_book_title_with_and_is_not_a_complex_question(corpus, monkeypatch):
    _use_llm(monkeypatch, ScriptedLLM())
    body = _query("Where did Phileas Fogg live and work?").json()
    assert body["route"] == "simple"


def test_weak_evidence_escalates_a_simple_question(corpus, monkeypatch):
    monkeypatch.setattr(settings, "evidence_min_dense_score", 0.99)
    scripted = ScriptedLLM(answer="I could not fully answer. Fogg lived in London [SOURCE 1].")
    _use_llm(monkeypatch, scripted)

    body = _query("Where did Phileas Fogg live?").json()

    assert scripted.decompositions == 1
    assert body["route"] == "complex"
    assert body["answer_status"] == "partial"
    assert len(scripted.contexts) == 1  # Decompose + one answer, no grading calls.


def test_provider_outage_serves_a_cited_extractive_answer(corpus, monkeypatch):
    _use_llm(monkeypatch, ScriptedLLM(error=httpx.ConnectError("down")))

    response = _query("Where did Phileas Fogg live?")

    assert response.status_code == 200
    body = response.json()
    assert body["answer_status"] == "extractive"
    assert body["fallback"] == "extractive"
    assert body["grounded"] is True
    assert "temporarily unavailable" in body["answer"]
    assert body["citations"][0]["source_uri"] == BOOK_A


def test_spent_daily_quota_serves_extractive_without_calling_a_provider(corpus, monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_token_budget", 10)
    quota.record(10)
    scripted = ScriptedLLM()
    _use_llm(monkeypatch, scripted)

    response = _query("Where did Phileas Fogg live?")

    assert response.status_code == 200
    assert response.json()["answer_status"] == "extractive"
    assert scripted.contexts == []


def test_fabricated_citation_is_removed_and_not_grounded(corpus, monkeypatch):
    _use_llm(monkeypatch, ScriptedLLM(answer="Fogg lived in Paris [SOURCE 9]."))

    body = _query("Where did Phileas Fogg live?").json()

    assert body["grounded"] is False
    assert body["citations"] == []
    assert "SOURCE 9" not in body["answer"]


def test_citation_to_the_wrong_chunk_is_not_grounded(corpus, monkeypatch):
    answer = "Zebras migrate across savannah grasslands annually [SOURCE 1]."
    _use_llm(monkeypatch, ScriptedLLM(answer=answer))

    body = _query("Where did Phileas Fogg live?").json()

    assert body["citations"]  # The marker resolves, but its claim has no support.
    assert body["grounded"] is False


def test_other_tenants_get_no_evidence_and_no_llm_call(corpus, monkeypatch):
    monkeypatch.setattr(settings, "auth_tenant_id", "tenant-other")
    scripted = ScriptedLLM()
    _use_llm(monkeypatch, scripted)

    body = _query("Where did Phileas Fogg live?").json()

    assert body["answer_status"] == "refused"
    assert body["grounded"] is False
    assert scripted.contexts == []


def test_metrics_report_the_query_without_its_text(corpus, monkeypatch):
    _use_llm(monkeypatch, ScriptedLLM())
    _query("Where did Phileas Fogg live?")

    response = TestClient(api.app).get("/metrics")

    assert response.status_code == 200
    assert 'rag_requests_total{method="POST",route="/query",status="200"}' in response.text
    assert 'rag_answer_status_total{status="ok"}' in response.text
    assert "rag_retrieval_top_dense_score_bucket" in response.text
    assert "Phileas" not in response.text
