"""Integration check that retrieval enforces tenant and role filters."""

import uuid
from unittest.mock import patch

import pytest
from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import Settings
from src.indexing.retriever import (
    RRF_K,
    HybridRetriever,
    build_hybrid_retriever,
    build_reranker,
)

# QueryFusionRetriever resolves the global LLM even for a single query.
LlamaSettings.llm = MockLLM()


def _node(text, tenant_id, roles):
    return TextNode(
        text=text,
        metadata={"tenant_id": tenant_id, "allowed_roles": roles},
        id_=str(uuid.uuid4()),
    )


def test_reranker_scores_only_the_passage_text_and_keeps_every_candidate():
    hits = [
        NodeWithScore(node=_node(text, "secret-tenant", ["public"]), score=0.1)
        for text in ("weak passage", "strong passage")
    ]
    with patch("sentence_transformers.CrossEncoder") as encoder:
        encoder.return_value.predict.return_value = [-2.0, 3.0]
        reranked = build_reranker().postprocess_nodes(hits, query_str="question")

    pairs = encoder.return_value.predict.call_args.args[0]
    assert pairs == [("question", "weak passage"), ("question", "strong passage")]
    assert [hit.node.get_content() for hit in reranked] == ["strong passage", "weak passage"]
    assert [hit.score for hit in reranked] == [3.0, -2.0]


def test_hybrid_retriever_only_returns_permitted_chunks():
    client = QdrantClient(":memory:")
    try:
        store = QdrantVectorStore(client=client, collection_name="acl-test")
        index = VectorStoreIndex.from_vector_store(
            store, embed_model=MockEmbedding(embed_dim=8)
        )
        nodes = [
            _node("public evidence", "tenant-demo", ["public"]),
            _node("admin-only evidence", "tenant-demo", ["admin"]),
            _node("other tenant evidence", "tenant-other", ["public"]),
        ]
        index.insert_nodes(nodes)

        visible = [
            node
            for node in nodes
            if node.metadata["tenant_id"] == "tenant-demo"
            and "public" in node.metadata["allowed_roles"]
        ]
        retriever = build_hybrid_retriever(
            index, visible, tenant_id="tenant-demo", roles=["public"]
        )

        hits = retriever.retrieve("evidence")

        assert {hit.node.get_content() for hit in hits} == {"public evidence"}
    finally:
        client.close()


class _StubRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query):
        return self._hits


def _hit(node_id, score):
    return NodeWithScore(node=TextNode(text=node_id, id_=node_id), score=score)


def test_fusion_ranks_by_agreement_not_by_raw_score():
    # BM25's scale would win a raw-score sort; RRF rewards the shared document.
    dense = _StubRetriever([_hit("shared", 0.8), _hit("dense-only", 0.7)])
    sparse = _StubRetriever([_hit("bm25-only", 11.0), _hit("shared", 9.0)])

    hits = HybridRetriever(dense, sparse, top_k=3).retrieve("q")

    assert [hit.node.node_id for hit in hits] == ["shared", "bm25-only", "dense-only"]
    assert hits[0].score == 1 / (RRF_K + 1) + 1 / (RRF_K + 2)


def test_fusion_keeps_the_best_dense_score_on_every_hit():
    dense = _StubRetriever([_hit("a", 0.42), _hit("b", 0.2)])
    sparse = _StubRetriever([_hit("c", 9.0)])

    hits = HybridRetriever(dense, sparse, top_k=2).retrieve("q")

    assert {hit.top_dense_score for hit in hits} == {0.42}


def test_fusion_ignores_zero_score_bm25_padding_and_truncates():
    dense = _StubRetriever([_hit("a", 0.5)])
    sparse = _StubRetriever([_hit("padding", 0.0), _hit("b", 2.0), _hit("c", 1.0)])

    hits = HybridRetriever(dense, sparse, top_k=2).retrieve("q")

    assert [hit.node.node_id for hit in hits] == ["a", "b"]


def test_fusion_without_dense_results_reports_no_dense_score():
    hits = HybridRetriever(_StubRetriever([]), _StubRetriever([_hit("b", 2.0)]), 2).retrieve("q")
    assert hits[0].top_dense_score is None


def test_bm25_ranking_keeps_bm25_order_and_the_dense_score():
    dense = _StubRetriever([_hit("shared", 0.42), _hit("dense-only", 0.3)])
    sparse = _StubRetriever([_hit("first", 11.0), _hit("shared", 9.0), _hit("third", 3.0)])

    hits = HybridRetriever(dense, sparse, top_k=2, ranking="bm25").retrieve("q")

    assert [hit.node.node_id for hit in hits] == ["first", "shared"]
    assert [hit.score for hit in hits] == [11.0, 9.0]
    assert {hit.top_dense_score for hit in hits} == {0.42}


def test_bm25_ranking_falls_back_to_dense_when_no_term_matches():
    dense = _StubRetriever([_hit("a", 0.5)])
    sparse = _StubRetriever([_hit("padding", 0.0)])

    hits = HybridRetriever(dense, sparse, top_k=2, ranking="bm25").retrieve("q")

    assert [hit.node.node_id for hit in hits] == ["a"]
    assert hits[0].top_dense_score == 0.5


def test_a_call_can_override_the_configured_ranking():
    dense = _StubRetriever([_hit("shared", 0.8), _hit("dense-only", 0.7)])
    sparse = _StubRetriever([_hit("bm25-only", 11.0), _hit("shared", 9.0)])
    retriever = HybridRetriever(dense, sparse, top_k=3, ranking="bm25")

    assert retriever.retrieve("q")[0].node.node_id == "bm25-only"
    assert retriever.retrieve("q", ranking="rrf")[0].node.node_id == "shared"


def test_profiles_choose_their_retrieval_ranking():
    assert Settings(_env_file=None, profile="laptop").retrieval_ranking == "bm25"
    assert Settings(_env_file=None, profile="server").retrieval_ranking == "rrf"
    explicit = Settings(_env_file=None, profile="server", retrieval_ranking="BM25")
    assert explicit.retrieval_ranking == "bm25"


def test_unknown_retrieval_ranking_is_rejected():
    with pytest.raises(ValueError, match="RETRIEVAL_RANKING"):
        Settings(_env_file=None, retrieval_ranking="dense")
