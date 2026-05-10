"""Document ingestion and management endpoints."""

import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.logging_config import get_logger
from app.core.security import verify_api_key
from app.schemas.document import IngestResponse, ReindexResponse
from app.services.ingestion_service import IngestionService

logger = get_logger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document",
    description="Upload and ingest a PDF, DOCX, TXT, or CSV file into the vector knowledge base.",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.INGEST_RATE_LIMIT)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="File to ingest (PDF, DOCX, TXT, CSV)"),
) -> IngestResponse:
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    # Validate extension
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(ext)

    # Read file
    file_bytes = await file.read()

    # Validate size
    if len(file_bytes) > settings.max_file_size_bytes:
        raise FileTooLargeError(settings.MAX_FILE_SIZE_MB)

    logger.info(
        "Document upload received",
        extra={"file_name": filename, "size_bytes": len(file_bytes), "ext": ext},
    )

    service = IngestionService(request.app.state.vector_store)
    return await service.ingest_document(file_bytes, filename)


@router.post(
    "/reindex",
    response_model=ReindexResponse,
    summary="Re-index documents",
    description="Trigger a re-indexing of all persisted documents.",
    dependencies=[Depends(verify_api_key)],
)
async def reindex_documents(request: Request) -> ReindexResponse:
    start = time.monotonic()
    service = IngestionService(request.app.state.vector_store)
    result = await service.reindex_all(request.app.state.vector_store)
    duration = round(time.monotonic() - start, 3)
    logger.info("Re-index triggered", extra={"duration_s": duration})
    return ReindexResponse(
        status=result["status"],
        documents_reindexed=result["documents_reindexed"],
        total_chunks=result["total_chunks"],
        duration_seconds=duration,
        message=result["message"],
    )
