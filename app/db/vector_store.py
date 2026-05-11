"""
Pinecone vector store — cloud-hosted, survives restarts.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:

    def __init__(self):
        self._index = None

    async def initialize(self) -> None:
        start = time.monotonic()
        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)

            existing = [i.name for i in pc.list_indexes()]
            if settings.PINECONE_INDEX_NAME not in existing:
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=3072,   # gemini-embedding-exp-03-07
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info("Pinecone index created")

            self._index = pc.Index(settings.PINECONE_INDEX_NAME)
            latency = (time.monotonic() - start) * 1000
            stats = self._index.describe_index_stats()
            logger.info(
                "Pinecone initialized",
                extra={
                    "total_vectors": stats.total_vector_count,
                    "latency_ms": round(latency, 2),
                },
            )
        except Exception as e:
            logger.error("Pinecone init failed", extra={"error": str(e)})
            raise VectorStoreError(f"Pinecone init failed: {e}")

    async def close(self) -> None:
        logger.info("Pinecone connection closed")

    def _ensure_ready(self):
        if self._index is None:
            raise VectorStoreError("Vector store not initialized")

    async def add_documents(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        self._ensure_ready()
        start = time.monotonic()
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

        try:
            vectors = []
            for i, chunk_id in enumerate(chunk_ids):
                vectors.append({
                    "id": chunk_id,
                    "values": embeddings[i],
                    "metadata": {**metadatas[i], "content": chunks[i]},
                })

            # Pinecone recommends batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                self._index.upsert(vectors=vectors[i:i + batch_size])

            latency = (time.monotonic() - start) * 1000
            logger.info(
                "Documents added to Pinecone",
                extra={
                    "document_id": document_id,
                    "num_chunks": len(chunks),
                    "latency_ms": round(latency, 2),
                },
            )
            return chunk_ids
        except Exception as e:
            logger.error("Pinecone upsert failed", extra={"error": str(e)})
            raise VectorStoreError(f"Failed to store embeddings: {e}")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_ready()
        start = time.monotonic()

        try:
            results = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )
            latency = (time.monotonic() - start) * 1000

            hits = []
            for match in results.matches:
                if match.score >= settings.VECTOR_SIMILARITY_THRESHOLD:
                    metadata = dict(match.metadata)
                    content = metadata.pop("content", "")
                    hits.append({
                        "chunk_id": match.id,
                        "content": content,
                        "metadata": metadata,
                        "relevance_score": round(match.score, 4),
                    })

            logger.info(
                "Pinecone search completed",
                extra={"hits": len(hits), "latency_ms": round(latency, 2)},
            )
            return hits
        except Exception as e:
            logger.error("Pinecone search failed", extra={"error": str(e)})
            raise VectorStoreError(f"Search failed: {e}")

    async def delete_document(self, document_id: str) -> int:
        self._ensure_ready()
        try:
            # Pinecone supports delete by metadata filter (paid) 
            # or by IDs on free tier
            stats_before = self._index.describe_index_stats()
            self._index.delete(filter={"document_id": document_id})
            stats_after = self._index.describe_index_stats()
            deleted = stats_before.total_vector_count - stats_after.total_vector_count
            logger.info("Document deleted", extra={"document_id": document_id})
            return deleted
        except Exception as e:
            raise VectorStoreError(f"Delete failed: {e}")

    async def get_all_document_ids(self) -> List[str]:
        self._ensure_ready()
        stats = self._index.describe_index_stats()
        return list(stats.namespaces.keys())

    def count(self) -> int:
        if self._index is None:
            return 0
        try:
            return self._index.describe_index_stats().total_vector_count
        except Exception:
            return 0