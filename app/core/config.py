"""
Configuration management using Pydantic Settings.
All settings are loaded from environment variables or .env file.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Security
    SECRET_KEY: str = Field(default="change-me-in-production-use-long-random-string")
    API_KEYS: str = Field(default="dev-key-1,dev-key-2")
    ALLOWED_ORIGINS: List[str] = ["*"]
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: str = "minute"
    CHAT_RATE_LIMIT: str = "20/minute"
    INGEST_RATE_LIMIT: str = "10/minute"

    # LLM (Gemini)
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_MAX_TOKENS: int = 2048
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_TIMEOUT: int = 30
    GEMINI_MAX_RETRIES: int = 3

    # Vector Database (Pinecone) 
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_INDEX_NAME: str = Field(default="rag-chatbot")
    VECTOR_SEARCH_TOP_K: int = 5
    VECTOR_SIMILARITY_THRESHOLD: float = 0.3

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 50
    UPLOAD_DIR: str = "./uploads"
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".csv"]

    # Chat History
    CHAT_HISTORY_DB_PATH: str = "./chat_history.db"
    MAX_HISTORY_TURNS: int = 10
    HISTORY_RETENTION_DAYS: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "./logs/app.log"
    LOG_MAX_BYTES: int = 10_485_760
    LOG_BACKUP_COUNT: int = 5

    # System Prompt
    SYSTEM_PROMPT: str = (
        "You are a helpful AI assistant with access to a knowledge base. "
        "Answer questions using the provided context. If the context doesn't "
        "contain enough information, say so clearly rather than making things up. "
        "Be concise, accurate, and cite relevant parts of the context when helpful."
    )
    
    

    @property
    def api_keys_list(self) -> List[str]:
        return [k.strip() for k in self.API_KEYS.split(",") if k.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
