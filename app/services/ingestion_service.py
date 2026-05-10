"""
Document ingestion orchestration service.
Coordinates text extraction, chunking, embedding, and vector storage.
"""

import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

from app.core.config import settings
from app.core.exceptions import DocumentIngestionError
from app.core.logging_config import get_logger
from app.db.vector_store import VectorStore
from app.schemas.document import IngestResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class IngestionService:
    """End-to-end document ingestion pipeline."""

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._processor = DocumentProcessor()
        self._embedder = EmbeddingService()

    async def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        extra_metadata: Dict[str, Any] = None,
    ) -> IngestResponse:
        """Full ingestion pipeline: parse → chunk → embed → store."""
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        pipeline_start = time.monotonic()

        logger.info(
            "Starting document ingestion",
            extra={"document_id": document_id, "file_name": filename, "file_type": file_type},
        )

        # 1. Extract text
        text = self._processor.extract_text(file_bytes, filename)
        if not text:
            raise DocumentIngestionError(f"No text could be extracted from '{filename}'.")

        # 2. Chunk
        chunks = self._processor.chunk_text(text)
        if not chunks:
            raise DocumentIngestionError(f"Document produced no text chunks.")

        # 3. Generate embeddings
        logger.info("Generating embeddings", extra={"document_id": document_id, "num_chunks": len(chunks)})
        embeddings = await self._embedder.embed_texts(chunks)

        # 4. Build metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "document_id": document_id,
                "source": filename,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(extra_metadata or {}),
            }
            metadatas.append(meta)

        # 5. Store in vector DB
        await self._vector_store.add_documents(document_id, chunks, embeddings, metadatas)

        total_latency = (time.monotonic() - pipeline_start) * 1000
        logger.info(
            "Document ingestion complete",
            extra={
                "document_id": document_id,
                "file_name": filename,
                "num_chunks": len(chunks),
                "total_latency_ms": round(total_latency, 2),
            },
        )

        return IngestResponse(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            num_chunks=len(chunks),
            status="success",
            ingested_at=datetime.utcnow(),
            message=f"Document '{filename}' ingested successfully with {len(chunks)} chunks.",
        )

    async def reindex_all(self, vector_store: VectorStore) -> Dict[str, Any]:
        """Placeholder for re-indexing from persisted sources."""
        logger.info("Re-index requested (documents already persisted in ChromaDB)")
        count = vector_store.count()
        return {
            "status": "success",
            "documents_reindexed": 0,
            "total_chunks": count,
            "message": "ChromaDB is persistent; all existing chunks are already indexed.",
        }
