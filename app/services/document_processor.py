"""
Document processing service.
Handles file parsing, text extraction, and chunking for PDF, DOCX, TXT, CSV.
"""

import csv
import io
import os
import time
from pathlib import Path
from typing import List, Tuple

from app.core.config import settings
from app.core.exceptions import UnsupportedFileTypeError, DocumentIngestionError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Parses and chunks uploaded documents."""

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract plain text from a file based on its extension."""
        ext = Path(filename).suffix.lower()
        start = time.monotonic()
        text = ""

        try:
            if ext == ".txt":
                text = file_bytes.decode("utf-8", errors="replace")
            elif ext == ".csv":
                text = self._extract_csv(file_bytes)
            elif ext == ".pdf":
                text = self._extract_pdf(file_bytes)
            elif ext == ".docx":
                text = self._extract_docx(file_bytes)
            else:
                raise UnsupportedFileTypeError(ext)

            latency = (time.monotonic() - start) * 1000
            logger.info(
                "Text extracted",
                extra={"file_name": filename, "ext": ext, "chars": len(text), "latency_ms": round(latency, 2)},
            )
            return text.strip()
        except (UnsupportedFileTypeError, DocumentIngestionError):
            raise
        except Exception as e:
            logger.error("Text extraction failed", extra={"file_name": filename, "error": str(e)})
            raise DocumentIngestionError(f"Failed to extract text from '{filename}': {e}")

    def _extract_csv(self, file_bytes: bytes) -> str:
        text_io = io.StringIO(file_bytes.decode("utf-8", errors="replace"))
        reader = csv.reader(text_io)
        rows = [", ".join(row) for row in reader]
        return "\n".join(rows)

    def _extract_pdf(self, file_bytes: bytes) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            return "\n\n".join(pages)
        except ImportError:
            raise DocumentIngestionError("pypdf not installed. Run: pip install pypdf")

    def _extract_docx(self, file_bytes: bytes) -> str:
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise DocumentIngestionError("python-docx not installed. Run: pip install python-docx")

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks by word boundary."""
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        if not text:
            return []

        # Word-aware chunking
        words = text.split()
        chunks = []
        start = 0

        # Approximate chars-per-word ratio
        avg_word_len = max(1, len(text) // len(words)) if words else 5

        chunk_words = chunk_size // avg_word_len
        overlap_words = overlap // avg_word_len

        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end == len(words):
                break
            start += chunk_words - overlap_words

        logger.debug(
            "Text chunked",
            extra={"total_words": len(words), "num_chunks": len(chunks), "chunk_size": chunk_size},
        )
        return chunks
