"""
Authentication and authorization utilities.
API key-based auth with dependency injection.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key from X-API-Key header."""
    if not api_key:
        logger.warning("Request missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Pass it in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if api_key not in settings.api_keys_list:
        logger.warning("Invalid API key attempt", extra={"key_prefix": api_key[:8]})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired API key.",
        )
    logger.debug("API key authenticated", extra={"key_prefix": api_key[:8]})
    return api_key
