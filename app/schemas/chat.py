"""Pydantic schemas for chat interactions."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class SourceDocument(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    relevance_score: float
    excerpt: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=20)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    include_sources: bool = True


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    query: str
    answer: str
    sources: List[SourceDocument] = []
    tokens_used: Optional[int] = None
    latency_ms: float
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    total_messages: int
    created_at: datetime
    last_updated: datetime
