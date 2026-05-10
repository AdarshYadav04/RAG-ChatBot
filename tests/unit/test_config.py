"""Unit tests for settings configuration."""

import pytest
from app.core.config import Settings


def test_default_settings():
    s = Settings()
    assert s.APP_VERSION == "1.0.0"
    assert s.ENVIRONMENT == "development"
    assert s.CHUNK_SIZE > 0
    assert s.CHUNK_OVERLAP < s.CHUNK_SIZE


def test_api_keys_list():
    s = Settings(API_KEYS="key1,key2,key3")
    keys = s.api_keys_list
    assert "key1" in keys
    assert "key2" in keys
    assert len(keys) == 3


def test_max_file_size_bytes():
    s = Settings(MAX_FILE_SIZE_MB=10)
    assert s.max_file_size_bytes == 10 * 1024 * 1024


def test_is_production():
    s = Settings(ENVIRONMENT="production")
    assert s.is_production is True
    s2 = Settings(ENVIRONMENT="development")
    assert s2.is_production is False
