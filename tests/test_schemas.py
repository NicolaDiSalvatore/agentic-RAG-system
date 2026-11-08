"""Tests for the public schemas and the ingestion manifest."""

import uuid
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import ValidationError

from src.indexing import build_index
from src.schemas import CanonicalDocument, Citation, IndexedChunk


def test_indexed_chunk_round_trips_and_defaults_roles():
    chunk = IndexedChunk(
        document_id="doc-1",
        chunk_id="chunk-1",
        text="evidence",
        section="Chapter 1",
        source_uri="books/a.txt",
        content_hash="sha256:abc",
        index_version="v1",
    )

    assert chunk.tenant_id == "tenant-demo"
    assert chunk.allowed_roles == ["public"]
    assert IndexedChunk.model_validate(chunk.model_dump()) == chunk


def test_canonical_document_requires_canonical_fields():
    with pytest.raises(ValidationError):
        CanonicalDocument(document_id="doc-1")

    document = CanonicalDocument(
        document_id="doc-1",
        source_uri="books/a.txt",
        title="A",
        text="full text",
        content_hash="sha256:abc",
        index_version="v1",
    )
    assert document.acl.allowed_roles == ["public"]


def test_citation_requires_document_chunk_and_uri():
    with pytest.raises(ValidationError):
        Citation(document_id="doc-1", chunk_id="chunk-1")

    citation = Citation(
        document_id="doc-1", chunk_id="chunk-1", source_uri="books/a.txt"
    )
    assert citation.model_dump()["chunk_id"] == "chunk-1"


def test_build_nodes_produces_schema_valid_metadata(tmp_path):
    source = tmp_path / "chunks.parquet"
    pd.DataFrame(
        {
            "Chunk": ["first chunk", "second chunk"],
            "Book Name": ["A_Book_-_An_Author"] * 2,
            "Chapter": [1, 2],
        }
    ).to_parquet(source, index=False)

    nodes = build_index._build_nodes(source)

    assert len(nodes) == 2
    for node in nodes:
        chunk = IndexedChunk.model_validate(
            {**node.metadata, "text": node.get_content()}
        )
        # Qdrant only accepts UUID point IDs; the content hash is the chunk ID.
        assert str(uuid.UUID(node.node_id)) == node.node_id
        assert node.node_id == build_index._point_id(chunk.chunk_id)
        assert chunk.content_hash.startswith("sha256:")
        assert chunk.source_uri == "A_Book_-_An_Author"
        assert chunk.tenant_id == "tenant-demo"


def test_collection_name_is_versioned_and_idempotent(monkeypatch):
    monkeypatch.setattr(build_index.settings, "qdrant_collection", "rag_system")
    monkeypatch.setattr(build_index.settings, "index_version", "v2")

    assert build_index._collection_name() == "rag_system__v2"

    monkeypatch.setattr(build_index.settings, "qdrant_collection", "rag_system__v2")
    assert build_index._collection_name() == "rag_system__v2"


def test_manifest_records_and_validates_ingestion(tmp_path, monkeypatch):
    source = tmp_path / "chunks.parquet"
    source.write_bytes(b"source-bytes")
    monkeypatch.setattr(build_index, "MANIFEST_ROOT", tmp_path / "manifests")
    monkeypatch.setattr(build_index, "_manifest_cache", None)
    monkeypatch.setattr(build_index.settings, "qdrant_collection", "rag_system")
    monkeypatch.setattr(build_index.settings, "index_version", "v3")
    monkeypatch.setattr(build_index.settings, "embedding_model", "model-a")

    build_index._write_manifest(source, point_count=42)

    manifest = build_index.read_manifest()
    assert manifest is not None
    assert manifest["point_count"] == 42
    assert manifest["index_version"] == "v3"
    assert manifest["collection_name"] == "rag_system__v3"
    assert build_index.active_index_version() == "v3"
    assert build_index._manifest_error(manifest) is None

    # Serving the same collection with a different embedding model must fail
    # closed instead of mixing incompatible vectors.
    monkeypatch.setattr(build_index.settings, "embedding_model", "model-b")
    error = build_index._manifest_error(build_index.read_manifest())
    assert error is not None and "model-b" in error


def test_index_is_ready_requires_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "MANIFEST_ROOT", tmp_path / "manifests")
    monkeypatch.setattr(build_index, "_manifest_cache", None)
    client = MagicMock()
    client.count.return_value.count = 5
    index = MagicMock()
    index.vector_store.client = client
    monkeypatch.setattr(build_index, "_cached_index", index)

    # Points exist, but without a manifest the collection is not trusted.
    assert build_index.index_is_ready() is False

    source = tmp_path / "chunks.parquet"
    source.write_bytes(b"source-bytes")
    build_index._write_manifest(source, point_count=5)

    assert build_index.index_is_ready() is True


def test_index_is_ready_local_mode_checks_persisted_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(build_index, "MANIFEST_ROOT", tmp_path / "manifests")
    monkeypatch.setattr(build_index, "_manifest_cache", None)
    monkeypatch.setattr(build_index, "_cached_index", None)
    monkeypatch.setattr(build_index, "QDRANT_PATH", tmp_path / "qdrant")
    monkeypatch.setattr(build_index.settings, "qdrant_url", "")

    source = tmp_path / "chunks.parquet"
    source.write_bytes(b"source-bytes")
    build_index._write_manifest(source, point_count=5)

    # Manifest alone is not enough: the collection storage must exist.
    assert build_index.index_is_ready() is False

    storage = (
        tmp_path
        / "qdrant"
        / "collection"
        / build_index._collection_name()
        / "storage.sqlite"
    )
    storage.parent.mkdir(parents=True)
    storage.write_bytes(b"points")

    assert build_index.index_is_ready() is True


def test_manifest_file_is_valid_json(tmp_path, monkeypatch):
    source = tmp_path / "chunks.parquet"
    source.write_bytes(b"source-bytes")
    monkeypatch.setattr(build_index, "MANIFEST_ROOT", tmp_path / "manifests")
    monkeypatch.setattr(build_index, "_manifest_cache", None)

    build_index._write_manifest(source, point_count=1)

    path = build_index._manifest_path()
    assert json.loads(path.read_text(encoding="utf-8"))["point_count"] == 1
