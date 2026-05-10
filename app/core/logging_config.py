"""
Production-ready logging configuration with JSON and text formatters,
rotating file handlers, and structured logging support.
"""

import logging
import logging.handlers
import os
import sys
from typing import Any, Dict

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        extra_keys = {
            k for k in record.__dict__
            if k not in logging.LogRecord.__dict__ and not k.startswith("_")
            and k not in ("msg", "args", "exc_info", "exc_text", "stack_info",
                          "levelno", "levelname", "pathname", "filename",
                          "module", "lineno", "funcName", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName",
                          "process", "name", "message", "taskName")
        }
        for key in extra_keys:
            log_data[key] = getattr(record, key)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_data["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_data, default=str)


class ColoredTextFormatter(logging.Formatter):
    """Colored text formatter for development/console output."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> None:
    """Configure logging for the application."""
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_FORMAT == "json":
        console_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        console_handler.setFormatter(
            ColoredTextFormatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialized",
        extra={"level": settings.LOG_LEVEL, "format": settings.LOG_FORMAT},
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
