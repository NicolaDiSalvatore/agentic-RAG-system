"""
Load pre-chunked documents, embed their text, and index them in Qdrant using LlamaIndex.
"""

from pathlib import Path
import atexit
import hashlib
import json
import os
import pickle
import uuid
from datetime import UTC, datetime
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from diskcache import Cache
from src.config import settings, get_embed_model
from src.schemas import IndexedChunk
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).parents[2]
QDRANT_PATH = PROJECT_ROOT / ".cache" / "qdrant"
NODE_CACHE_ROOT = PROJECT_ROOT / ".cache" / "corpus_nodes"
EMBED_CACHE_ROOT = PROJECT_ROOT / ".cache" / "corpus_embeddings"
MANIFEST_ROOT = PROJECT_ROOT / ".cache" / "manifests"
COLLECTION_NAME = "rag_system"

EMBED_BATCH_SIZE = 64
INSERT_CHUNK_SIZE = 2048

_cached_index: VectorStoreIndex | None = None
_cached_nodes: list[TextNode] | None = None
_embed_cache: "Cache | None" = None
_manifest_cache: dict | None = None


def _collection_name() -> str:
    """Return the active Qdrant collection name from settings.

    ``settings.qdrant_collection`` is the single source of truth so embedded
    and remote modes share one name; ``COLLECTION_NAME`` is the fallback for
    misconfigured environments.

    Returns:
        The Qdrant collection name to read from and write to.
    """
    base = settings.qdrant_collection or COLLECTION_NAME
    suffix = f"__{settings.index_version}"
    return base if base.endswith(suffix) else f"{base}{suffix}"


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
    # Keep the identity compatible with the original ingestion format. New
    # governance metadata (tenant, ACL, versions, and the derived chunk ID)
    # must not create duplicate vectors for an otherwise unchanged chunk.
    meta = {
        str(k): (str(v) if v is not None else None)
        for k, v in sorted(metadata.items(), key=lambda pair: str(pair[0]))
        if str(k) in {"file_name", "chapter"}
    }
    payload = json.dumps(
        {"text": text, "metadata": meta},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _point_id(content_key: str) -> str:
    """Derive a stable Qdrant point ID, which must be a UUID, from a content hash."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{content_key}"))


def _build_nodes(source_path: Path) -> list[TextNode]:
    """Build TextNodes from the pre-chunked parquet file.

    Args:
        source_path: Path to the chunked parquet source file.

    Returns:
        A list of TextNodes whose ``chunk_id`` metadata is the content hash, so
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
    # embedding each Document independently. chunk_id is the content hash so
    # incremental builds can detect and reuse already-indexed chunks.
    nodes = []
    for row in dataset:
        text = str(row["Chunk"] or "")
        if not text.strip():
            continue
        source_uri = str(row["Book Name"])
        chapter = str(row["Chapter"])
        document_id = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:24]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # The legacy identity fields stay part of the hash so existing
        # embeddings remain reusable; governance metadata is derived, not hashed.
        content_key = _content_key(text, {"file_name": source_uri, "chapter": chapter})
        chunk = IndexedChunk(
            document_id=document_id,
            chunk_id=content_key,
            text=text,
            section=chapter,
            source_uri=source_uri,
            tenant_id="tenant-demo",
            allowed_roles=["public"],
            content_hash=f"sha256:{text_hash}",
            index_version=settings.index_version,
        )
        metadata = chunk.model_dump(exclude={"text"})
        # Keep the original field names the retriever and prompts already use.
        metadata["file_name"] = source_uri
        metadata["chapter"] = chapter
        # ``node_id`` is a read-only property; the constructor kwarg is ``id_``.
        nodes.append(TextNode(text=text, metadata=metadata, id_=_point_id(content_key)))
    return nodes


def _manifest_path(collection: str | None = None) -> Path:
    """Return the manifest path for a collection (defaults to the active one)."""
    return MANIFEST_ROOT / f"{collection or _collection_name()}.json"


def read_manifest(collection: str | None = None) -> dict | None:
    """Read the ingestion manifest, returning ``None`` when it is absent.

    The manifest is the only record of which embedding model and index version
    produced a collection, so callers use it to fail closed when serving an
    index that does not match the current configuration.
    """
    global _manifest_cache
    name = collection or _collection_name()
    if _manifest_cache is not None and _manifest_cache.get("collection_name") == name:
        return _manifest_cache
    try:
        manifest = json.loads(_manifest_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    _manifest_cache = manifest
    return manifest


def _manifest_error(manifest: dict | None) -> str | None:
    """Return why ``manifest`` cannot serve the current settings, if it cannot."""
    if manifest is None:
        return (
            "No ingestion manifest was found for this collection. "
            "Run `python -m scripts.ingest --path data/gutenqa_chunks.parquet` first."
        )
    if manifest.get("collection_name") != _collection_name():
        return (
            f"Manifest collection '{manifest.get('collection_name')}' does not match "
            f"the configured collection '{_collection_name()}'."
        )
    if manifest.get("embedding_model") != settings.embedding_model:
        return (
            "The index was built with embedding model "
            f"'{manifest.get('embedding_model')}' but EMBEDDING_MODEL is "
            f"'{settings.embedding_model}'. Re-ingest under a new INDEX_VERSION."
        )
    if manifest.get("index_version") != settings.index_version:
        return (
            f"The index version '{manifest.get('index_version')}' does not match "
            f"INDEX_VERSION '{settings.index_version}'."
        )
    if not manifest.get("point_count"):
        return "The ingestion manifest records no indexed points."
    return None


def active_index_version() -> str:
    """Return the index version recorded by the manifest, or the configured one."""
    manifest = read_manifest()
    if manifest and manifest.get("index_version"):
        return str(manifest["index_version"])
    return settings.index_version


def _write_manifest(source_path: Path, point_count: int) -> None:
    """Persist a local, auditable record of a completed ingestion run."""
    global _manifest_cache
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_revision": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "parser_version": "parquet-gutenqa-v1",
        "chunker_version": "upstream-prechunked-v1",
        "embedding_model": settings.embedding_model,
        "collection_name": _collection_name(),
        "index_version": settings.index_version,
        "point_count": point_count,
        "created_at": datetime.now(UTC).isoformat(),
    }
    target = _manifest_path()
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    _manifest_cache = manifest


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


def _embed_nodes(nodes: list[TextNode], batch_size: int | None = None) -> None:
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
    batch_size = batch_size or settings.embedding_batch_size or EMBED_BATCH_SIZE

    embedder = get_embed_model()
    cache = _get_embed_cache()

    missed = [node for node in nodes if node.metadata["chunk_id"] not in cache]
    with tqdm(total=len(nodes), desc="Embedding", unit="chunk") as pbar:
        for start in range(0, len(missed), batch_size):
            batch = missed[start : start + batch_size]
            vectors = embedder._embed([node.text for node in batch])
            for node, vector in zip(batch, vectors):
                node.embedding = vector
                cache[node.metadata["chunk_id"]] = vector
            pbar.update(len(batch))
        for node in nodes:
            if node.embedding is None:
                node.embedding = cache[node.metadata["chunk_id"]]
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


def _node_cache_path(collection: str | None = None) -> Path:
    """Return the BM25 node cache file for a collection."""
    return NODE_CACHE_ROOT / f"{collection or _collection_name()}.pkl"


def _node_cache_identity(point_count: int) -> dict:
    """Describe the exact index a node cache must belong to.

    A point count alone cannot identify a corpus: two index versions, or the
    same corpus before and after a metadata change, can hold identical counts.
    Reusing nodes from the wrong one would serve stale text to BM25, or drop
    everything at the tenant filter when the old nodes lack governance metadata.

    Args:
        point_count: The number of points currently in the collection.

    Returns:
        The values that must all match for a cached node list to be reused.
    """
    manifest = read_manifest() or {}
    return {
        "collection_name": _collection_name(),
        "index_version": settings.index_version,
        "embedding_model": settings.embedding_model,
        "source_revision": manifest.get("source_revision"),
        "parser_version": manifest.get("parser_version"),
        "chunker_version": manifest.get("chunker_version"),
        "point_count": point_count,
    }


def _load_node_cache(identity: dict) -> list[TextNode] | None:
    """Load pickled corpus nodes when they belong to exactly this index.

    Args:
        identity: The expected value of ``_node_cache_identity``.

    Returns:
        The cached node list if a matching cache exists, otherwise ``None``.
    """
    path = _node_cache_path(identity["collection_name"])
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            cached = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(cached, dict) or cached.get("identity") != identity:
        return None
    nodes = cached.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != identity["point_count"]:
        return None
    return nodes


def _save_node_cache(nodes: list[TextNode], identity: dict) -> None:
    """Persist corpus nodes so later runs skip the Qdrant payload scroll.

    Writes atomically via a temporary file so a crash mid-write never leaves
    a corrupt cache behind.

    Args:
        nodes: The corpus nodes to persist.
        identity: The value of ``_node_cache_identity`` the nodes belong to.

    Returns:
        None.
    """
    path = _node_cache_path(identity["collection_name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {"identity": identity, "nodes": nodes},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def load_corpus_nodes(index: VectorStoreIndex) -> list[TextNode]:
    """Recover all chunk nodes from the Qdrant store without re-embedding.

    Qdrant keeps node text in its payloads and leaves the index docstore
    empty, so BM25 needs this scroll-based reconstruction instead of
    ``index.docstore.docs``. A pickle cache tied to the index's identity avoids
    rescrolling every payload on each process start.

    Args:
        index: The vector index whose Qdrant store holds the corpus.

    Returns:
        The list of all TextNodes stored in the collection, loaded from the
            pickle cache when it belongs to this exact index.
    """
    global _cached_nodes
    if _cached_nodes is not None:
        return _cached_nodes

    client: QdrantClient = index.vector_store.client
    collection = _collection_name()
    expected_count = client.count(collection_name=collection, exact=True).count

    identity = _node_cache_identity(expected_count)
    cached = _load_node_cache(identity)
    if cached is not None:
        _cached_nodes = cached
        return _cached_nodes

    nodes: list[TextNode] = []
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
        _save_node_cache(nodes, identity)
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
    global _cached_index, _cached_nodes
    if _cached_index is not None:
        return _cached_index

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Chunk data file not found: {source}")

    _configure_embed_model()
    vector_store = _ensure_collection()
    collection = _collection_name()
    existing_count = vector_store.client.count(collection_name=collection, exact=True).count
    existing_keys = _existing_content_keys(vector_store) if existing_count else set()

    nodes = _build_nodes(source)
    missing = [node for node in nodes if node.metadata["chunk_id"] not in existing_keys]
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
    _write_manifest(source, final_count)
    _cached_nodes = None
    _cached_index = VectorStoreIndex.from_vector_store(vector_store)
    return _cached_index


def build_index(path: str) -> VectorStoreIndex:
    """Build the configured collection from the corpus, resuming if interrupted.

    This delegates to the incremental ``fill_index`` for both storage modes. A
    partial collection is finished rather than accepted, and only the
    configured collection is touched: other index versions stay available so
    switching ``INDEX_VERSION`` back remains a working rollback.

    Args:
        path: Path to the chunked parquet source file.

    Returns:
        A VectorStoreIndex backed by the Qdrant collection, either remote
            (``settings.qdrant_url`` set) or embedded local (empty URL).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    return fill_index(path)


def load_existing_index() -> VectorStoreIndex:
    """Open a non-empty persisted collection without indexing documents."""
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    _configure_embed_model()
    client = _open_client()
    collection = _collection_name()
    if not client.collection_exists(collection):
        client.close()
        raise RuntimeError(
            f"Qdrant collection '{collection}' does not exist. "
            "Run `python -m scripts.ingest --path data/gutenqa_chunks.parquet` first."
        )
    count = client.count(collection_name=collection, exact=True).count
    if count < 1:
        client.close()
        raise RuntimeError(
            f"Qdrant collection '{collection}' is empty. Run ingestion before starting the API."
        )
    problem = _manifest_error(read_manifest())
    if problem:
        client.close()
        raise RuntimeError(problem)
    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    _cached_index = VectorStoreIndex.from_vector_store(vector_store)
    return _cached_index


def index_is_ready() -> bool:
    """Return whether the configured collection is trustworthy and has data.

    Readiness must never raise and must never block: in embedded local mode,
    opening a Qdrant client loads the whole collection (tens of seconds for a
    large corpus) and takes the storage lock, so the probe inspects the
    manifest and the collection's on-disk storage instead. Remote mode counts
    points over the network.
    """
    if _manifest_error(read_manifest()) is not None:
        return False
    try:
        if _cached_index is not None:
            client = _cached_index.vector_store.client
            return client.count(collection_name=_collection_name(), exact=True).count > 0
        if not settings.qdrant_url:
            storage = QDRANT_PATH / "collection" / _collection_name() / "storage.sqlite"
            return storage.is_file() and storage.stat().st_size > 0
        client = _open_client()
        try:
            collection = _collection_name()
            return client.collection_exists(collection) and (
                client.count(collection_name=collection, exact=True).count > 0
            )
        finally:
            client.close()
    except Exception:
        return False


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
