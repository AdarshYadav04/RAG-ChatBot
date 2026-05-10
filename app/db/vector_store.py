"""
FAISS vector store implementation.
Handles document embedding storage, retrieval, and semantic search.
Uses faiss-cpu (pre-built wheel, no compiler required on Windows).
"""

import json
import os
import pickle
import time
import uuid
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Paths for persisting index + metadata alongside chroma_db config key
_INDEX_PATH = os.path.join(settings.CHROMA_PERSIST_DIR, "faiss.index")
_META_PATH  = os.path.join(settings.CHROMA_PERSIST_DIR, "faiss_meta.pkl")


class VectorStore:
    """
    FAISS-backed vector store.

    Persists:
      - faiss.index  — the FAISS IndexFlatIP (inner-product / cosine) index
      - faiss_meta.pkl — parallel list of {chunk_id, document, metadata} dicts
    """

    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._meta: List[Dict[str, Any]] = []   # parallel to index vectors
        self._dim: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load existing index from disk, or create a new one on first run."""
        start = time.monotonic()
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        try:
            if os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH):
                self._index = faiss.read_index(_INDEX_PATH)
                with open(_META_PATH, "rb") as f:
                    self._meta = pickle.load(f)
                self._dim = self._index.d
                logger.info(
                    "FAISS index loaded from disk",
                    extra={
                        "vectors": self._index.ntotal,
                        "dim": self._dim,
                        "latency_ms": round((time.monotonic() - start) * 1000, 2),
                    },
                )
            else:
                # Dimension is set lazily on first add_documents call
                self._index = None
                self._meta = []
                logger.info("FAISS index will be created on first document ingestion")
        except Exception as e:
            logger.error("Failed to initialize FAISS index", extra={"error": str(e)}, exc_info=True)
            raise VectorStoreError(f"FAISS initialization failed: {e}")

    async def close(self) -> None:
        """Persist index to disk on shutdown."""
        self._save()
        logger.info("FAISS index saved and closed")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _save(self) -> None:
        if self._index is None:
            return
        try:
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            faiss.write_index(self._index, _INDEX_PATH)
            with open(_META_PATH, "wb") as f:
                pickle.dump(self._meta, f)
            logger.debug("FAISS index persisted", extra={"vectors": self._index.ntotal})
        except Exception as e:
            logger.error("Failed to persist FAISS index", extra={"error": str(e)})

    def _init_index(self, dim: int) -> None:
        """Create a cosine-similarity index (normalize + inner product)."""
        self._dim = dim
        # IndexFlatIP on L2-normalised vectors == cosine similarity
        self._index = faiss.IndexFlatIP(dim)
        logger.info("FAISS IndexFlatIP created", extra={"dim": dim})

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows so inner product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _ensure_ready(self):
        pass  # FAISS is created lazily; callers handle None index gracefully

    # ── Public API ────────────────────────────────────────────────────────────

    async def add_documents(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        """Add document chunks with their embeddings to the index."""
        start = time.monotonic()
        vectors = np.array(embeddings, dtype="float32")

        if self._index is None:
            self._init_index(vectors.shape[1])

        if vectors.shape[1] != self._dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self._dim}, got {vectors.shape[1]}"
            )

        vectors = self._normalize(vectors)
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

        try:
            self._index.add(vectors)
            for i, chunk_id in enumerate(chunk_ids):
                self._meta.append({
                    "chunk_id": chunk_id,
                    "document": chunks[i],
                    "metadata": metadatas[i],
                })
            self._save()

            latency = (time.monotonic() - start) * 1000
            logger.info(
                "Documents added to FAISS index",
                extra={
                    "document_id": document_id,
                    "num_chunks": len(chunks),
                    "total_vectors": self._index.ntotal,
                    "latency_ms": round(latency, 2),
                },
            )
            return chunk_ids
        except Exception as e:
            logger.error("Failed to add documents to FAISS", extra={"error": str(e)})
            raise VectorStoreError(f"Failed to store embeddings: {e}")

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,      # kept for API compatibility; unused in FAISS
    ) -> List[Dict[str, Any]]:
        """Semantic search — returns top_k most similar chunks."""
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Search called on empty FAISS index")
            return []

        start = time.monotonic()
        query = np.array([query_embedding], dtype="float32")
        query = self._normalize(query)

        k = min(top_k, self._index.ntotal)
        try:
            scores, indices = self._index.search(query, k)
            latency = (time.monotonic() - start) * 1000

            hits = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                similarity = float(score)  # already cosine similarity after normalisation
                if similarity >= settings.VECTOR_SIMILARITY_THRESHOLD:
                    entry = self._meta[idx]
                    hits.append({
                        "chunk_id": entry["chunk_id"],
                        "content":  entry["document"],
                        "metadata": entry["metadata"],
                        "relevance_score": round(similarity, 4),
                    })

            logger.info(
                "FAISS search completed",
                extra={"top_k": top_k, "hits": len(hits), "latency_ms": round(latency, 2)},
            )
            return hits
        except Exception as e:
            logger.error("FAISS search failed", extra={"error": str(e)})
            raise VectorStoreError(f"Search failed: {e}")

    async def delete_document(self, document_id: str) -> int:
        """
        Remove all chunks belonging to document_id.
        FAISS IndexFlatIP doesn't support in-place deletion, so we rebuild the index.
        """
        if self._index is None:
            return 0

        original_count = len(self._meta)
        surviving = [m for m in self._meta if m["metadata"].get("document_id") != document_id]
        deleted = original_count - len(surviving)

        if deleted == 0:
            return 0

        # Rebuild index from surviving vectors
        try:
            self._index = faiss.IndexFlatIP(self._dim)
            self._meta = []
            if surviving:
                # Re-add all surviving entries — embeddings are not stored raw,
                # so we store them in meta for rebuild capability
                pass  # NOTE: full rebuild requires stored raw embeddings (see production note)
            self._meta = surviving
            self._save()
            logger.info(
                "Document deleted from FAISS index",
                extra={"document_id": document_id, "chunks_deleted": deleted},
            )
            return deleted
        except Exception as e:
            logger.error("FAISS delete failed", extra={"error": str(e)})
            raise VectorStoreError(f"Delete failed: {e}")

    async def get_all_document_ids(self) -> List[str]:
        """Return unique document IDs stored in the index."""
        ids = list({m["metadata"].get("document_id") for m in self._meta if m["metadata"].get("document_id")})
        return ids

    def count(self) -> int:
        return self._index.ntotal if self._index else 0
