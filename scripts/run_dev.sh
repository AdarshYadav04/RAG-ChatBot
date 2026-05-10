#!/usr/bin/env bash
set -e
mkdir -p logs uploads chroma_db
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
