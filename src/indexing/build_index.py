"""
Load pre-chunked documents, embed their text, and index them in Qdrant using LlamaIndex.
"""

from pathlib import Path
import atexit
import hashlib
import json
import os
import pickle
import shutil
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from diskcache import Cache
from src.config import settings, get_embed_model
from tqdm import tqdm
from src.config import get_embed_model


PROJECT_ROOT = Path(__file__).parents[2]
QDRANT_PATH = PROJECT_ROOT / ".cache" / "qdrant"
NODE_CACHE_PATH = PROJECT_ROOT / ".cache" / "corpus_nodes.pkl"
EMBED_CACHE_ROOT = PROJECT_ROOT / ".cache" / "corpus_embeddings"
COLLECTION_NAME = "rag_system"

EMBED_BATCH_SIZE = 128
INSERT_CHUNK_SIZE = 2048

_cached_index: VectorStoreIndex | None = None
_cached_nodes: list[TextNode] | None = None
_embed_cache: "Cache | None" = None


def _collection_name() -> str:
    """Return the active Qdrant collection name from settings.

    ``settings.qdrant_collection`` is the single source of truth so embedded
    and remote modes share one name; ``COLLECTION_NAME`` is the fallback for
    misconfigured environments.

    Returns:
        The Qdrant collection name to read from and write to.
    """
    return settings.qdrant_collection or COLLECTION_NAME


def _open_client() -> QdrantClient:
    """Open a Qdrant client for the configured storage mode.

    An empty ``settings.qdrant_url`` selects Qdrant's embedded local mode
    over ``QDRANT_PATH``; a non-empty URL targets a remote Qdrant server
    (e.g. the docker-compose service) with optional API-key auth.

    Returns:
        A QdrantClient bound to either the local storage folder or the
            remote server.
    """
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return QdrantClient(path=str(QDRANT_PATH))


def _configure_embed_model() -> None:
    """Configure global LlamaIndex settings with the app's embed model.

    The retrieval pipeline never generates text through LlamaIndex, but
    QueryFusionRetriever resolves ``Settings.llm`` at construction time, so a
    MockLLM is installed to avoid the OpenAI-default lookup and the API key it
    would otherwise require.

    Returns:
        None.
    """
    Settings.embed_model = get_embed_model()
    # QueryFusionRetriever resolves Settings.llm on construction even though
    # this pipeline never generates through LlamaIndex; MockLLM avoids the
    # OpenAI-default lookup and its required API key.
    Settings.llm = MockLLM()


def _content_key(text: str, metadata: dict) -> str:
    """Compute a stable content hash used as both node_id and cache key.

    Args:
        text: The chunk text.
        metadata: A dict of associated metadata values. Values from the
            parquet can be int/float/str across runs, so everything is
            coerced to its string representation to keep the key stable.

    Returns:
        A SHA-256 hex digest of the canonicalized text/metadata payload.
    """
    meta = {
        str(k): (str(v) if v is not None else None)
        for k, v in sorted(metadata.items(), key=lambda pair: str(pair[0]))
    }
    payload = json.dumps(
        {"text": text, "metadata": meta},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_nodes(source_path: Path) -> list[TextNode]:
    """Build TextNodes from the pre-chunked parquet file.

    Args:
        source_path: Path to the chunked parquet source file.

    Returns:
        A list of TextNodes with node_id set to the content hash so
        incremental builds can detect and reuse already-indexed chunks.

    Raises:
        ValueError: If the dataset is missing any of the required columns
            (``Chunk``, ``Book Name``, ``Chapter``).
    """
    dataset = load_dataset("parquet", data_files=str(source_path), split="train")
    required_columns = {"Chunk", "Book Name", "Chapter"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Chunk data is missing required columns: {sorted(missing_columns)}")
    # The parquet file is already chunked. Construct nodes directly so
    # VectorStoreIndex embeds them in batches instead of transforming and
    # embedding each Document independently. node_id is the content hash so
    # incremental builds can detect and reuse already-indexed chunks.
    nodes = []
    for row in dataset:
        text = str(row["Chunk"] or "")
        if not text.strip():
            continue
        metadata = {"file_name": row["Book Name"], "chapter": row["Chapter"]}
        node_id = _content_key(text, metadata)
        nodes.append(TextNode(text=text, metadata=metadata, node_id=node_id))
    return nodes


def _get_embed_cache() -> "Cache":
    """Return the per-model disk embedding cache.

    Each embedding model family gets its own directory under
    ``EMBED_CACHE_ROOT`` so caches never collide across model switches.

    Returns:
        The diskcache.Cache instance for the active embedding model.
    """
    global _embed_cache
    if _embed_cache is None:
        model_dir = EMBED_CACHE_ROOT / settings.embedding_model.replace("/", "__")
        model_dir.mkdir(parents=True, exist_ok=True)
        _embed_cache = Cache(str(model_dir))
    return _embed_cache


def _embed_nodes(nodes: list[TextNode], batch_size: int = EMBED_BATCH_SIZE) -> None:
    """Embed nodes in batches, persisting vectors to the disk cache.

    Fills ``node.embedding`` for every node, preferring the cached vector
    when the content hash was embedded by an earlier run so already-embedded
    chunks are not re-embedded. Unfinished batches from interrupted runs are
    resumed in place.

    Args:
        nodes: The nodes to embed in place.
        batch_size: Maximum number of nodes embedded per model call.

    Returns:
        None. Mutates ``nodes`` by filling each node's ``embedding``.
    """
    if not nodes:
        return

    embedder = get_embed_model()
    cache = _get_embed_cache()

    missed = [node for node in nodes if node.node_id not in cache]
    with tqdm(total=len(nodes), desc="Embedding", unit="chunk") as pbar:
        for start in range(0, len(missed), batch_size):
            batch = missed[start : start + batch_size]
            vectors = embedder._embed([node.text for node in batch])
            for node, vector in zip(batch, vectors):
                node.embedding = vector
                cache[node.node_id] = vector
            pbar.update(len(batch))
        for node in nodes:
            if node.embedding is None:
                node.embedding = cache[node.node_id]
        pbar.close()


def _insert_missing(vector_store: QdrantVectorStore, nodes: list[TextNode]) -> None:
    """Upsert embedded nodes into the store in fixed-size chunks.

    Args:
        vector_store: The destination Qdrant vector store.
        nodes: The embedded nodes to add.

    Returns:
        None. Nodes are added to ``vector_store``; nodes without a vector
            are silently skipped by the underlying upsert.
    """
    for start in range(0, len(nodes), INSERT_CHUNK_SIZE):
        vector_store.add(nodes[start: start + INSERT_CHUNK_SIZE])


def _ensure_collection() -> QdrantVectorStore:
    """Open (creating if needed) the Qdrant vector store.

    Creates the collection on first run using the fixed 384-dimensional
    cosine configuration expected by the embedding model. In local mode the
    storage directory is created on demand; in remote mode the server-side
    collection is created if missing.

    Returns:
        A QdrantVectorStore bound to the configured collection.
    """
    collection = _collection_name()
    client = _open_client()
    if not settings.qdrant_url:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=collection)


def _existing_content_keys(vector_store: QdrantVectorStore) -> set[str]:
    """Return the set of content hashes already present in the collection.

    Args:
        vector_store: The Qdrant vector store to inspect.

    Returns:
        A set of content hashes for every stored point, reconstructed from
        its ``_node_content`` payload.
    """
    keys: set[str] = set()
    client = vector_store.client
    collection = _collection_name()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            raw = point.payload.get("_node_content")
            if not raw:
                continue
            node = TextNode.model_validate_json(raw)
            keys.add(_content_key(node.text, node.metadata or {}))
        if offset is None:
            break
    return keys


def _load_node_cache(expected_count: int) -> list[TextNode] | None:
    """Load pickled corpus nodes when they match the current point count.

    Args:
        expected_count: The number of points currently in the collection;
            a mismatched count invalidates the cache.

    Returns:
        The cached node list if a valid cache exists, otherwise ``None``.
    """
    if not NODE_CACHE_PATH.is_file():
        return None
    try:
        with NODE_CACHE_PATH.open("rb") as handle:
            cached = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(cached, list) or len(cached) != expected_count:
        return None
    return cached


def _save_node_cache(nodes: list[TextNode]) -> None:
    """Persist corpus nodes so later runs skip the Qdrant payload scroll.

    Writes atomically via a temporary file so a crash mid-write never leaves
    a corrupt cache behind.

    Args:
        nodes: The corpus nodes to persist.

    Returns:
        None.
    """
    NODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = NODE_CACHE_PATH.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, NODE_CACHE_PATH)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def load_corpus_nodes(index: VectorStoreIndex) -> list[TextNode]:
    """Recover all chunk nodes from the Qdrant store without re-embedding.

    Qdrant keeps node text in its payloads and leaves the index docstore
    empty, so BM25 needs this scroll-based reconstruction instead of
    ``index.docstore.docs``. A pickle cache keyed by point count avoids
    rescrolling every payload on each process start.

    Args:
        index: The vector index whose Qdrant store holds the corpus.

    Returns:
        The list of all TextNodes stored in the collection, loaded from the
            pickle cache when it matches the current point count.
    """
    global _cached_nodes
    if _cached_nodes is not None:
        return _cached_nodes

    client: QdrantClient = index.vector_store.client
    collection = _collection_name()
    expected_count = client.count(collection_name=collection, exact=True).count

    cached = _load_node_cache(expected_count)
    if cached is not None:
        _cached_nodes = cached
        return _cached_nodes

    nodes = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        nodes.extend(TextNode.model_validate_json(point.payload["_node_content"]) for point in points)
        if offset is None:
            break

    if len(nodes) == expected_count:
        _save_node_cache(nodes)
    _cached_nodes = nodes
    return _cached_nodes


def fill_index(source_path: str, expected_count: int | None = None) -> VectorStoreIndex:
    """Incrementally add corpus chunks to the persisted Qdrant collection.

    Unlike a wipe-and-rebuild, this never destroys or re-embeds content that
    already exists: chunks whose content hash is present in the collection or
    in the on-disk embedding cache are skipped, so interrupted builds resume
    in place instead of restarting from zero.

    Args:
        source_path: Path to the chunked parquet source file.
        expected_count: Optional expected final point count asserted after
            the build completes.

    Returns:
        A VectorStoreIndex backed by the persisted Qdrant collection.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        AssertionError: If ``expected_count`` is given and does not match the
            final collection point count.
    """
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Chunk data file not found: {source_path}")

    _configure_embed_model()
    vector_store = _ensure_collection()
    collection = _collection_name()
    existing_count = vector_store.client.count(collection_name=collection, exact=True).count
    existing_keys = _existing_content_keys(vector_store) if existing_count else set()

    nodes = _build_nodes(source_path)
    missing = [node for node in nodes if node.node_id not in existing_keys]
    if missing:
        _embed_nodes(missing)
        _insert_missing(vector_store, missing)

    final_count = vector_store.client.count(
        collection_name=_collection_name(), exact=True
    ).count
    if expected_count is not None:
        assert final_count == expected_count, (
            f"Collection has {final_count} points, expected {expected_count}; "
            f"{len(missing)} chunks were missing and newly embedded."
        )
    _cached_index = VectorStoreIndex.from_vector_store(vector_store)
    return _cached_index


def build_index(path: str) -> VectorStoreIndex:
    """Build or reuse the Qdrant index for the given corpus.

    Prefers the existing collection when it already has points rather than
    wiping and re-embedding the corpus (a full embed is a costly CPU-only
    operation). The local storage folder is only wiped when the collection is
    missing or empty, because QdrantLocal's delete_collection leaves stale
    points in its in-memory cache; a remote server collection is never wiped.

    Args:
        path: Path to the chunked parquet source file.

    Returns:
        A VectorStoreIndex backed by the Qdrant collection, either remote
            (``settings.qdrant_url`` set) or embedded local (empty URL).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Chunk data file not found: {source_path}")

    # Both paths need the embed model: rebuilds embed the corpus, and the
    # fast path embeds queries at retrieval time.
    _configure_embed_model()

    collection = _collection_name()

    if settings.qdrant_url:
        client = _open_client()
        if client.collection_exists(collection):
            qdrant_count = client.count(collection_name=collection, exact=True).count
            # Load whatever exists rather than wiping and re-embedding on a
            # count mismatch: a partial collection is still usable, and the
            # full corpus embed is a costly CPU-only operation.
            if qdrant_count > 0:
                vector_store = QdrantVectorStore(client=client, collection_name=collection)
                _cached_index = VectorStoreIndex.from_vector_store(vector_store)
                return _cached_index
        client.close()
        # Remote server storage has no QdrantLocal in-memory-cache caveat: a
        # missing or empty collection is safely created and filled in place.
        return fill_index(str(source_path))

    if QDRANT_PATH.exists():
        client = _open_client()
        if client.collection_exists(collection):
            qdrant_count = client.count(collection_name=collection, exact=True).count
            # Load whatever exists rather than wiping and re-embedding on a
            # count mismatch: a partial collection is still usable, and the
            # full corpus embed is a costly CPU-only operation. The storage
            # folder is only wiped when the collection is missing or empty.
            if qdrant_count > 0:
                vector_store = QdrantVectorStore(client=client, collection_name=collection)
                _cached_index = VectorStoreIndex.from_vector_store(vector_store)
                return _cached_index
        client.close()
        # QdrantLocal's delete_collection leaves stale points in its in-memory
        # cache, so a same-name recreate resurrects old vectors; wiping the
        # storage folder is the only clean reset for local mode.
        shutil.rmtree(QDRANT_PATH)

    return fill_index(str(source_path))


def close_index() -> None:
    """Close the Qdrant client behind the cached index.

    QdrantClient.__del__ also calls close() during interpreter shutdown,
    when the import system is already torn down and ``import portalocker``
    inside QdrantLocal.close fails with ``ImportError: sys.meta_path is
    None``. Closing the client here while imports still work makes that
    final shutdown-time close a no-op (the flock file is already released)
    and silences the noise.

    Returns:
        None.
    """
    index = _cached_index
    if index is None:
        return
    vector_store = getattr(index, "vector_store", None)
    client = getattr(vector_store, "client", None) if vector_store is not None else None
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass


atexit.register(close_index)
