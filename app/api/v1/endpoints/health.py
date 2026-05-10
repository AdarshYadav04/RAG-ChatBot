"""Health check and status endpoints."""

import time

from fastapi import APIRouter, Request

from app.core.config import settings
from app.core.logging_config import get_logger
from app.schemas.health import ComponentStatus, HealthResponse

logger = get_logger(__name__)
router = APIRouter()
_START_TIME = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of all system components.",
)
async def health_check(request: Request) -> HealthResponse:
    components = {}

    # Vector store check
    vs_start = time.monotonic()
    try:
        vector_store = request.app.state.vector_store
        count = vector_store.count()
        components["vector_store"] = ComponentStatus(
            status="healthy",
            latency_ms=round((time.monotonic() - vs_start) * 1000, 2),
            details=f"{count} chunks indexed (FAISS)",
        )
    except Exception as e:
        components["vector_store"] = ComponentStatus(status="unhealthy", details=str(e))

    # Chat DB check
    db_start = time.monotonic()
    try:
        _ = request.app.state.chat_db
        components["chat_db"] = ComponentStatus(
            status="healthy",
            latency_ms=round((time.monotonic() - db_start) * 1000, 2),
            details="SQLite (aiosqlite)",
        )
    except Exception as e:
        components["chat_db"] = ComponentStatus(status="unhealthy", details=str(e))

    all_healthy = all(c.status == "healthy" for c in components.values())
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        components=components,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
    )
