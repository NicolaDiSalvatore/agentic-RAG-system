"""Stable public schemas shared by ingestion, retrieval, and the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AccessControl(BaseModel):
    allowed_roles: list[str] = Field(default_factory=lambda: ["public"])


class CanonicalDocument(BaseModel):
    document_id: str
    tenant_id: str = "tenant-demo"
    source_uri: str
    title: str
    text: str
    content_hash: str
    acl: AccessControl = Field(default_factory=AccessControl)
    index_version: str


class IndexedChunk(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    section: str | None = None
    source_uri: str
    tenant_id: str = "tenant-demo"
    allowed_roles: list[str] = Field(default_factory=lambda: ["public"])
    content_hash: str
    index_version: str


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    source_uri: str
