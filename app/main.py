"""
RAG Chatbot - Production-Ready FastAPI Application
Main application entry point with middleware, routing, and lifecycle management.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1.endpoints import chat, documents, health
from app.core.config import settings
from app.core.logging_config import get_logger, setup_logging
from app.db.vector_store import VectorStore
from app.db.chat_history import ChatHistoryDB

setup_logging()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown events."""
    logger.info("Starting RAG Chatbot application", extra={"event": "startup"})

    try:
        # Initialize vector store
        vector_store = VectorStore()
        await vector_store.initialize()
        app.state.vector_store = vector_store
        logger.info("Vector store initialized successfully")

        # Initialize chat history DB
        chat_db = ChatHistoryDB()
        await chat_db.initialize()
        app.state.chat_db = chat_db
        logger.info("Chat history database initialized successfully")

        logger.info(
            "Application startup complete",
            extra={"event": "startup_complete", "environment": settings.ENVIRONMENT},
        )
        yield
    except Exception as e:
        logger.critical(
            "Fatal error during startup", extra={"error": str(e)}, exc_info=True
        )
        raise
    finally:
        logger.info("Shutting down RAG Chatbot application", extra={"event": "shutdown"})
        if hasattr(app.state, "vector_store"):
            await app.state.vector_store.close()
        if hasattr(app.state, "chat_db"):
            await app.state.chat_db.close()
        logger.info("Shutdown complete")


def create_application() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Production-ready RAG Chatbot with semantic search and LLM integration",
        version=settings.APP_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID and latency logging middleware
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.monotonic()

        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        try:
            response: Response = await call_next(request)
            latency_ms = (time.monotonic() - start_time) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"

            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "path": request.url.path,
                },
            )
            return response
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "latency_ms": round(latency_ms, 2),
                    "path": request.url.path,
                },
                exc_info=True,
            )
            raise

    # Routers
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

    # Custom Swagger UI (protected in production)
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        return get_swagger_ui_html(
            openapi_url="/api/v1/openapi.json",
            title=f"{settings.APP_NAME} - API Docs",
            swagger_favicon_url="",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url="/api/v1/openapi.json",
            title=f"{settings.APP_NAME} - ReDoc",
        )

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=settings.APP_NAME,
            version=settings.APP_VERSION,
            description="RAG Chatbot API with document ingestion, semantic search, and LLM-powered responses.",
            routes=app.routes,
        )
        schema["components"]["securitySchemes"] = {
            "APIKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        }
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app


app = create_application()
