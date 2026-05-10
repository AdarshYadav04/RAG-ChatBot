"""
RAG orchestration service.
Retrieves context, generates answers, and persists chat history.
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.core.logging_config import get_logger
from app.db.chat_history import ChatHistoryDB
from app.db.vector_store import VectorStore
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse, SourceDocument
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class RAGService:
    """Retrieval-Augmented Generation orchestration."""

    def __init__(self, vector_store: VectorStore, chat_db: ChatHistoryDB):
        self._vector_store = vector_store
        self._chat_db = chat_db
        self._embedder = EmbeddingService()
        self._llm = LLMService()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        pipeline_start = time.monotonic()

        # Resolve or create session
        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        await self._chat_db.create_session(session_id)

        logger.info(
            "Processing chat request",
            extra={"session_id": session_id, "query_length": len(request.query)},
        )

        # 1. Load conversation history
        history = await self._chat_db.get_history(session_id, limit=settings.MAX_HISTORY_TURNS * 2)

        # 2. Embed query
        query_embedding = await self._embedder.embed_query(request.query)

        # 3. Semantic retrieval
        top_k = request.top_k or settings.VECTOR_SEARCH_TOP_K
        context_chunks = await self._vector_store.search(query_embedding, top_k=top_k)

        logger.info(
            "Context retrieved",
            extra={"session_id": session_id, "context_chunks": len(context_chunks)},
        )

        # 4. Generate answer
        answer, tokens_used = await self._llm.generate_answer(
            query=request.query,
            context_chunks=context_chunks,
            history=history,
            temperature=request.temperature,
        )

        # 5. Persist turn
        now = datetime.utcnow()
        new_messages = [
            ChatMessage(role="user", content=request.query, timestamp=now),
            ChatMessage(role="assistant", content=answer, timestamp=now),
        ]
        await self._chat_db.add_messages(session_id, new_messages)

        # 6. Build source list
        sources = []
        if request.include_sources:
            for chunk in context_chunks:
                meta = chunk.get("metadata", {})
                sources.append(
                    SourceDocument(
                        document_id=meta.get("document_id", "unknown"),
                        filename=meta.get("source", "unknown"),
                        chunk_id=chunk["chunk_id"],
                        relevance_score=chunk["relevance_score"],
                        excerpt=chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
                    )
                )

        total_latency = (time.monotonic() - pipeline_start) * 1000
        logger.info(
            "Chat response ready",
            extra={
                "session_id": session_id,
                "message_id": message_id,
                "tokens_used": tokens_used,
                "latency_ms": round(total_latency, 2),
            },
        )

        return ChatResponse(
            session_id=session_id,
            message_id=message_id,
            query=request.query,
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
            latency_ms=round(total_latency, 2),
            timestamp=now,
        )
