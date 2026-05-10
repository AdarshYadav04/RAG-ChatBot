"""Pydantic schemas for document ingestion and management."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str
    file_type: str
    page: Optional[int] = None
    chunk_index: int
    total_chunks: int
    extra: Dict[str, Any] = {}


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    num_chunks: int
    status: str
    ingested_at: datetime
    message: str


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    file_type: str
    num_chunks: int
    ingested_at: datetime
    status: str


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentListItem]


class ReindexResponse(BaseModel):
    status: str
    documents_reindexed: int
    total_chunks: int
    duration_seconds: float
    message: str
