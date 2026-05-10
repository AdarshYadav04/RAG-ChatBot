"""
Embedding generation service using Google Gemini.
"""

import time
from typing import List

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    BATCH_SIZE = 20  # Gemini embedding batch limit

    def __init__(self):
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.GEMINI_EMBEDDING_MODEL

    @retry(
        stop=stop_after_attempt(settings.GEMINI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        start = time.monotonic()
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        latency = (time.monotonic() - start) * 1000
        logger.debug(
            "Embedding batch generated",
            extra={"batch_size": len(texts), "model": self._model, "latency_ms": round(latency, 2)},
        )
        return [e.values for e in response.embeddings]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            try:
                embeddings = await self._embed_batch(batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error("Embedding generation failed", extra={"error": str(e)}, exc_info=True)
                raise VectorStoreError(f"Embedding generation failed: {e}")
        logger.info("Embeddings generated", extra={"total_texts": len(texts)})
        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        start = time.monotonic()
        response = self._client.models.embed_content(
            model=self._model,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        latency = (time.monotonic() - start) * 1000
        logger.debug("Query embedded", extra={"latency_ms": round(latency, 2)})
        return response.embeddings[0].values