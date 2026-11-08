"""Hybrid retrieval + reranking, built from LlamaIndex components.
"""

from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever
from src.config import settings
from llama_index.core.postprocessor import SentenceTransformerRerank


PROJECT_ROOT = Path(__file__).parents[2]

# The constant from the original RRF paper; it damps the gap between top ranks
# so one retriever's first hit cannot outweigh agreement between both retrievers.
RRF_K = 60


class FusedHit(NodeWithScore):
    """A ranked hit whose ``score`` orders results but cannot judge the evidence.

    ``score`` is a BM25 or RRF score. RRF scores depend only on ranks, and BM25
    scores off-topic queries about as highly as real ones. ``top_dense_score``
    carries the best cosine similarity seen for the query, which is the signal
    that can judge the evidence.
    """

    top_dense_score: float | None = None


class HybridRetriever:
    """Run dense and BM25 search, then rank by BM25 or Reciprocal Rank Fusion.

    Both searches always run, because the dense score judges the evidence even
    when BM25 alone orders the results. Fusing on ranks is what makes the two
    retrievers comparable: BM25 scores are unbounded (3-11 on this corpus) while
    cosine similarity cannot exceed 1, so sorting on raw scores would discard
    nearly every dense result.
    """

    def __init__(
        self, dense_retriever, sparse_retriever, top_k: int, ranking: str = "rrf"
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._top_k = top_k
        self._ranking = ranking

    def retrieve(self, query: str, ranking: str | None = None) -> list[FusedHit]:
        """Return the top hits for ``query``, best first.

        Args:
            query: The search text.
            ranking: ``"bm25"`` or ``"rrf"``, overriding the configured ranking.
                Calibration uses this to compare them on one retriever.

        Returns:
            The ranked hits. A query matching no BM25 term falls through to
            fusion, which then contains only the dense results.
        """
        dense_hits = list(self._dense.retrieve(query))
        # BM25 pads its results with zero-score documents when few terms match.
        sparse_hits = [hit for hit in self._sparse.retrieve(query) if (hit.score or 0.0) > 0.0]

        top_dense = max((hit.score for hit in dense_hits if hit.score is not None), default=None)
        if (ranking or self._ranking) == "bm25" and sparse_hits:
            return [
                FusedHit(node=hit.node, score=hit.score, top_dense_score=top_dense)
                for hit in sparse_hits[: self._top_k]
            ]

        fused: dict[str, tuple[float, BaseNode]] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits, start=1):
                score, _ = fused.get(hit.node.node_id, (0.0, hit.node))
                fused[hit.node.node_id] = (score + 1.0 / (RRF_K + rank), hit.node)

        ranked = sorted(fused.values(), key=lambda item: item[0], reverse=True)
        return [
            FusedHit(node=node, score=score, top_dense_score=top_dense)
            for score, node in ranked[: self._top_k]
        ]


def build_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    *,
    tenant_id: str | None = None,
    roles: list[str] | None = None,
) -> HybridRetriever:
    """Build a retriever running dense and BM25 search, ranked as configured.

    Each sub-retriever fetches a generous candidate pool (``top_k_dense`` /
    ``top_k_sparse``) and the ranking keeps ``top_k_dense`` results so the
    cross-encoder reranker has a real pool to reorder instead of being limited
    to the handful of top hits.

    Args:
        index: The vector index whose dense retriever participates in the fusion.
        nodes: Corpus nodes fed to the BM25 sparse retriever. Callers pass only
            the nodes the requesting identity may see.
        tenant_id: When set, the dense query is filtered by tenant in Qdrant.
        roles: When set with ``tenant_id``, the dense query is additionally
            filtered to chunks whose ``allowed_roles`` contain one of these.

    Returns:
        A retriever returning one ranked list that carries the dense score.
    """
    dense_kwargs: dict = {"similarity_top_k": settings.top_k_dense}
    if tenant_id:
        filters = [MetadataFilter(key="tenant_id", value=tenant_id)]
        if roles:
            filters.append(
                MetadataFilter(key="allowed_roles", value=list(roles), operator="in")
            )
        dense_kwargs["filters"] = MetadataFilters(filters=filters)
    dense_retriever = index.as_retriever(**dense_kwargs)

    # BM25 is rebuilt per process (the module-level retriever cache in
    # nodes.py means this happens once per identity). It cannot be cached to
    # disk: the default PyStemmer english stemmer is not picklable, so a
    # pickled retriever cannot be byte-serialized/reloaded.
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=settings.top_k_sparse
    )

    return HybridRetriever(
        dense_retriever,
        bm25_retriever,
        settings.top_k_dense,
        ranking=settings.retrieval_ranking,
    )


class TextOnlyRerank(SentenceTransformerRerank):
    """Rerank on passage text alone.

    The base class also scores each node's metadata. Here that is IDs, hashes and
    access fields, which add noise and use up the model's 512-token window.
    """

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if not nodes:
            return []
        pairs = [(query_bundle.query_str, hit.node.get_content()) for hit in nodes]
        for hit, score in zip(nodes, self._model.predict(pairs)):
            hit.score = float(score)
        return sorted(nodes, key=lambda hit: hit.score or 0.0, reverse=True)[: self.top_n]


def build_reranker() -> TextOnlyRerank:
    """Build the cross-encoder reranker configured from project settings.

    Every candidate stays in the result, reordered. The retrieve node applies the
    chunk budget, which grows on a retry and would otherwise find nothing new.

    Returns:
        A TextOnlyRerank using the configured reranker model.
    """
    return TextOnlyRerank(model=settings.reranker_model, top_n=settings.top_k_dense)
