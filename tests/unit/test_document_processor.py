"""Unit tests for DocumentProcessor."""

import pytest
from app.services.document_processor import DocumentProcessor
from app.core.exceptions import UnsupportedFileTypeError


@pytest.fixture
def processor():
    return DocumentProcessor()


def test_extract_txt(processor):
    content = b"Hello, world! This is a test document."
    result = processor.extract_text(content, "test.txt")
    assert "Hello" in result
    assert len(result) > 0


def test_extract_csv(processor):
    content = b"name,age,city\nAlice,30,NYC\nBob,25,LA"
    result = processor.extract_text(content, "data.csv")
    assert "Alice" in result
    assert "Bob" in result


def test_unsupported_extension(processor):
    with pytest.raises(UnsupportedFileTypeError):
        processor.extract_text(b"data", "file.xyz")


def test_chunk_text_basic(processor):
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = processor.chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_empty_text(processor):
    chunks = processor.chunk_text("")
    assert chunks == []


def test_chunk_short_text(processor):
    text = "This is a short document."
    chunks = processor.chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_overlap(processor):
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = processor.chunk_text(text, chunk_size=100, overlap=20)
    # With overlap, adjacent chunks should share words
    assert len(chunks) >= 2
