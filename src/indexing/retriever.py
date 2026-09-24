"""Hybrid retrieval + reranking, built from LlamaIndex components.
"""

from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from src.config import settings
from llama_index.core.postprocessor import SentenceTransformerRerank


PROJECT_ROOT = Path(__file__).parents[2]


def build_hybrid_retriever(index: VectorStoreIndex, nodes: list[BaseNode]) -> QueryFusionRetriever:
    """Build a hybrid retriever fusing dense vector and BM25 sparse retrieval.

    Combines the vector-store dense retriever with a BM25 sparse retriever
    over the given corpus nodes and fuses their results. Fusion runs
    synchronously because local-disk Qdrant clients expose no async API,
    so the fusion retriever cannot fan out to the sub-retrievers
    asynchronously.

    Each sub-retriever fetches a generous candidate pool (``top_k_dense`` /
    ``top_k_sparse``) and the fusion retriever keeps ``top_k_dense`` fused
    results so the cross-encoder reranker has a real pool to reorder instead
    of being limited to the handful of top fused hits.

    Args:
        index: The vector index whose dense retriever participates in the fusion.
        nodes: Corpus nodes fed to the BM25 sparse retriever.

    Returns:
        A QueryFusionRetriever that fuses dense and sparse hits into a single ranked result list.
    """
    dense_retriever = index.as_retriever(similarity_top_k=settings.top_k_dense)

    # BM25 is rebuilt per process (the module-level retriever cache in
    # nodes.py means this happens once per process). It cannot be cached to
    # disk: the default PyStemmer english stemmer is not picklable, so a
    # pickled retriever cannot be byte-serialized/reloaded.
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=settings.top_k_sparse)

    # Local-disk Qdrant clients expose no async API, so fusion must fan out
    # to the sub-retrievers synchronously.
    return QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=settings.top_k_dense,
        num_queries=1,
        use_async=False,
    )


def build_reranker() -> SentenceTransformerRerank:
    """Build the cross-encoder reranker configured from project settings.

    Returns:
        A SentenceTransformerRerank using the configured reranker model,
            keeping the top ``top_k_reranked`` passages after reranking.
    """
    return SentenceTransformerRerank(model=settings.reranker_model, top_n=settings.top_k_reranked)
