"""
LLM service for response generation using Google Gemini.
"""

import time
from typing import List, Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import LLMError
from app.core.logging_config import get_logger
from app.schemas.chat import ChatMessage

logger = get_logger(__name__)


class LLMService:
    def __init__(self):
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.GEMINI_MODEL

    def _build_context_prompt(self, query: str, context_chunks: List[dict]) -> str:
        if not context_chunks:
            return (
                f"The user asked: {query}\n\n"
                "No relevant context was found in the knowledge base. "
                "Please let the user know and answer from general knowledge if possible."
            )
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            context_parts.append(f"[Source {i}: {source}]\n{chunk['content']}")
        context_text = "\n\n---\n\n".join(context_parts)
        return (
            f"Use the following context to answer the question.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "Answer based on the context above. If the context is insufficient, say so."
        )

    def _build_history(self, history: List[ChatMessage]) -> List[types.Content]:
        contents = []
        for msg in history[-(settings.MAX_HISTORY_TURNS * 2):]:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        return contents

    @retry(
        stop=stop_after_attempt(settings.GEMINI_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    def _call_llm(self, contents: List[types.Content], system_prompt: str, temperature: float):
        return self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=settings.GEMINI_MAX_TOKENS,
                temperature=temperature,
            ),
        )

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[dict],
        history: List[ChatMessage],
        temperature: Optional[float] = None,
    ) -> tuple[str, int]:
        temp = temperature if temperature is not None else settings.GEMINI_TEMPERATURE
        history_contents = self._build_history(history)
        current_prompt = self._build_context_prompt(query, context_chunks)
        contents = history_contents + [
            types.Content(role="user", parts=[types.Part(text=current_prompt)])
        ]
        start = time.monotonic()

        try:
            response = self._call_llm(contents, settings.SYSTEM_PROMPT, temp)
            latency = (time.monotonic() - start) * 1000
            answer = response.text or ""
            tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0

            logger.info(
                "Gemini response generated",
                extra={"model": self._model, "tokens": tokens, "latency_ms": round(latency, 2)},
            )
            return answer, tokens
        except Exception as e:
            logger.error("LLM generation failed", extra={"error": str(e)}, exc_info=True)
            raise LLMError(f"Gemini error: {e}")