"""Chat and history endpoints."""

from fastapi import APIRouter, Depends, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.core.logging_config import get_logger
from app.core.security import verify_api_key
from app.schemas.chat import ChatRequest, ChatResponse, ChatHistoryResponse
from app.services.rag_service import RAGService

logger = get_logger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with your documents",
    description="Ask a question and receive a RAG-powered answer grounded in your knowledge base.",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    logger.info("Chat request received", extra={"session_id": body.session_id, "query": body.query[:80]})
    service = RAGService(request.app.state.vector_store, request.app.state.chat_db)
    return await service.chat(body)


@router.get(
    "/history/{session_id}",
    response_model=ChatHistoryResponse,
    summary="Retrieve chat history",
    description="Fetch conversation history for a given session ID.",
    dependencies=[Depends(verify_api_key)],
)
async def get_chat_history(request: Request, session_id: str) -> ChatHistoryResponse:
    chat_db = request.app.state.chat_db
    if not await chat_db.session_exists(session_id):
        raise SessionNotFoundError(session_id)
    messages = await chat_db.get_history(session_id)
    info = await chat_db.get_session_info(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_messages=info["total_messages"],
        created_at=info["created_at"],
        last_updated=info["last_updated"],
    )
