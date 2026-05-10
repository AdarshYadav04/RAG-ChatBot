# 🤖 RAG Chatbot — Production-Ready

A fully production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with **FastAPI**, **ChromaDB**, and **OpenAI**. Supports multi-format document ingestion, semantic search, conversation history, authentication, rate limiting, and full Docker support.

---

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py                    # FastAPI app factory, middleware, lifespan
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── chat.py        # POST /chat, GET /chat/history/{id}
│   │           ├── documents.py   # POST /documents/ingest, /reindex
│   │           └── health.py     # GET /health
│   ├── core/
│   │   ├── config.py             # Pydantic settings (env-driven)
│   │   ├── exceptions.py         # Custom HTTP exceptions
│   │   ├── logging_config.py     # JSON + rotating file logging
│   │   └── security.py           # API key authentication
│   ├── db/
│   │   ├── vector_store.py       # ChromaDB operations
│   │   └── chat_history.py       # SQLite async chat history
│   ├── schemas/
│   │   ├── chat.py               # ChatRequest, ChatResponse, etc.
│   │   ├── document.py           # IngestResponse, DocumentList, etc.
│   │   └── health.py             # HealthResponse
│   └── services/
│       ├── document_processor.py # Parse + chunk PDF/DOCX/TXT/CSV
│       ├── embedding_service.py  # OpenAI embeddings with batching
│       ├── ingestion_service.py  # End-to-end ingestion pipeline
│       ├── llm_service.py        # OpenAI chat with retry logic
│       └── rag_service.py        # RAG orchestration
├── tests/
│   ├── unit/
│   │   ├── test_document_processor.py
│   │   └── test_config.py
│   └── integration/
│       └── test_api_health.py
├── scripts/
│   ├── generate_api_key.py
│   └── run_dev.sh
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## 💻 Windows Compatibility

This project uses **faiss-cpu** for the vector store, which ships as a pre-built wheel — no Microsoft C++ Build Tools required. Simply `pip install -r requirements.txt` on Windows and it works out of the box.

---

## ✨ Features

| Category | Details |
|---|---|
| **APIs** | Document ingestion, Chat, History retrieval, Health check, Re-indexing |
| **File support** | PDF, DOCX, TXT, CSV |
| **Vector DB** | FAISS-CPU (persistent, cosine similarity, no compiler needed) |
| **LLM** | OpenAI GPT-4o-mini (configurable) |
| **Embeddings** | OpenAI text-embedding-3-small with batching |
| **Auth** | API key (`X-API-Key` header) |
| **Rate limiting** | Per-endpoint limits via SlowAPI |
| **Logging** | JSON structured logs + rotating file handler |
| **Retry logic** | Tenacity-powered exponential backoff for OpenAI |
| **History** | SQLite async session-based conversation history |
| **Docs** | Swagger UI at `/docs`, ReDoc at `/redoc` |
| **Docker** | Multi-stage Dockerfile + docker-compose |
| **Tests** | Unit + integration with pytest-cov |

---

## 🚀 Local Setup

### Prerequisites
- Python 3.12+
- OpenAI API key

### 1. Clone & Install

```bash
git clone <repo-url>
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-your-key-here
API_KEYS=my-secret-api-key
SECRET_KEY=your-random-secret
```

### 3. Generate API Keys (optional utility)

```bash
python scripts/generate_api_key.py           # API key
python scripts/generate_api_key.py --type secret-key  # Secret key
```

### 4. Run

```bash
make dev
# or
bash scripts/run_dev.sh
```

The server starts at `http://localhost:8000`.  
- Swagger UI: `http://localhost:8000/docs`  
- Health check: `http://localhost:8000/api/v1/health`

---

## 🐳 Docker Setup

### Build & Run

```bash
# Copy and configure env
cp .env.example .env
# Edit .env with your OpenAI key and API keys

# Build and start
make docker-up

# View logs
docker-compose logs -f

# Stop
make docker-down
```

Data is persisted in named Docker volumes:
- `chroma_data` — vector embeddings
- `chat_db` — conversation history
- `logs` — application logs
- `uploads` — uploaded files

---

## 🌐 API Reference

All protected endpoints require the header: `X-API-Key: your-api-key`

### Health Check
```http
GET /api/v1/health
```
**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "components": {
    "vector_store": { "status": "healthy", "latency_ms": 2.1, "details": "42 chunks indexed" },
    "chat_db": { "status": "healthy", "latency_ms": 0.5 }
  },
  "uptime_seconds": 3600.2
}
```

---

### Ingest a Document
```http
POST /api/v1/documents/ingest
X-API-Key: your-api-key
Content-Type: multipart/form-data

file: <PDF | DOCX | TXT | CSV>
```
**Response (201):**
```json
{
  "document_id": "doc_abc123def456",
  "filename": "product_manual.pdf",
  "file_type": "pdf",
  "num_chunks": 42,
  "status": "success",
  "ingested_at": "2025-01-15T10:30:00",
  "message": "Document 'product_manual.pdf' ingested successfully with 42 chunks."
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@/path/to/document.pdf"
```

---

### Chat
```http
POST /api/v1/chat/
X-API-Key: your-api-key
Content-Type: application/json

{
  "query": "What are the key features of the product?",
  "session_id": "session_xyz789",
  "top_k": 5,
  "include_sources": true
}
```
**Response (200):**
```json
{
  "session_id": "session_xyz789",
  "message_id": "msg_a1b2c3d4",
  "query": "What are the key features of the product?",
  "answer": "Based on the provided documentation, the key features include...",
  "sources": [
    {
      "document_id": "doc_abc123",
      "filename": "product_manual.pdf",
      "chunk_id": "doc_abc123_chunk_3",
      "relevance_score": 0.923,
      "excerpt": "Key features include automated processing, real-time..."
    }
  ],
  "tokens_used": 512,
  "latency_ms": 1240.5,
  "timestamp": "2025-01-15T10:31:00"
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the main topics", "session_id": "my-session"}'
```

---

### Chat History
```http
GET /api/v1/chat/history/{session_id}
X-API-Key: your-api-key
```
**Response:**
```json
{
  "session_id": "session_xyz789",
  "messages": [
    { "role": "user", "content": "What are the key features?", "timestamp": "2025-01-15T10:31:00" },
    { "role": "assistant", "content": "The key features include...", "timestamp": "2025-01-15T10:31:02" }
  ],
  "total_messages": 2,
  "created_at": "2025-01-15T10:31:00",
  "last_updated": "2025-01-15T10:31:02"
}
```

---

### Re-index
```http
POST /api/v1/documents/reindex
X-API-Key: your-api-key
```
**Response:**
```json
{
  "status": "success",
  "documents_reindexed": 0,
  "total_chunks": 42,
  "duration_seconds": 0.012,
  "message": "ChromaDB is persistent; all existing chunks are already indexed."
}
```

---

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run only unit tests
pytest tests/unit/ -v

# Run with HTML report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

---

## 🔧 Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** — OpenAI API key |
| `API_KEYS` | `dev-key-1,dev-key-2` | Comma-separated valid API keys |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `VECTOR_SEARCH_TOP_K` | `5` | Context chunks per query |
| `VECTOR_SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload size |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns retained |
| `CHAT_RATE_LIMIT` | `20/minute` | Chat endpoint rate limit |
| `INGEST_RATE_LIMIT` | `10/minute` | Ingest endpoint rate limit |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ENVIRONMENT` | `development` | `development\|staging\|production` |

---

## 🏭 Production Deployment

### Checklist
- [ ] Set `ENVIRONMENT=production`
- [ ] Use a strong random `SECRET_KEY` (`make generate-secret`)
- [ ] Rotate `API_KEYS` to strong values (`make generate-key`)
- [ ] Set `ALLOWED_ORIGINS` to your domain(s)
- [ ] Set `DEBUG=false`
- [ ] Configure external log aggregation (ship `./logs/app.log`)
- [ ] Mount volumes on persistent storage
- [ ] Set up an Nginx reverse proxy with TLS termination
- [ ] Configure horizontal scaling (adjust `WORKERS`)

### Scaling Considerations
- **Multiple replicas**: ChromaDB persists to disk — use a shared volume or migrate to Qdrant/Weaviate with a dedicated server for multi-replica setups.
- **Async throughput**: Uvicorn with multiple workers handles concurrent requests; each worker maintains its own ChromaDB connection.
- **Embedding costs**: Batch embedding during ingestion; cache embeddings for frequently queried terms.

---

## 🛠️ Linting & Formatting

```bash
make lint      # ruff + mypy
make format    # ruff format
```

---

## 📄 License

MIT
