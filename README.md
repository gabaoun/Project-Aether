# Project Aether

Event-driven **Retrieval-Augmented Generation (RAG)** search engine built on Python and LlamaIndex Workflows. Aether ingests documents through a privacy-first pipeline (PII masking → chunking → LLM metadata enrichment → hybrid vector storage), then answers queries through a high-precision retrieval stack: semantic caching, HyDE, relevance judgment with refinement loops, and cross-encoder reranking.

Aether is served through a **FastAPI** REST interface, an interactive **CLI**, and asynchronous **background workers**, with a resilient degraded mode that keeps the system operational when external infrastructure fails.

---

## Key Capabilities

- **Event-Driven Ingestion:** LlamaIndex `Workflow` orchestration — documents are loaded, masked for PII, chunked, metadata-enriched by an LLM, and indexed into Chroma Cloud asynchronously (RQ background jobs persisted in PostgreSQL).
- **Hybrid Vector Search:** Chroma Cloud with dense (Qwen) + sparse (Splade) embeddings and RRF fusion for recall across both lexical and semantic matching.
- **High-Precision Retrieval:** HyDE (Hypothetical Document Embeddings), multi-hop query refinement with relevance judgment, `LongContextReorder`, and `BAAI/bge-reranker-v2-m3` cross-encoder reranking.
- **Semantic Caching:** Redis-backed cache keyed by embedding similarity (threshold-configurable, default `0.85`) that bypasses LLM computation for semantically identical queries — cutting token cost and latency.
- **PII Compliance Layer:** Regex-based masker sanitizes emails and phone numbers before documents enter the index.
- **Degraded Mode Resilience:** Redis and vector-store connection failures are detected at startup and at runtime — caching and search degrade gracefully instead of crashing the service.
- **Fault Tolerance:** Retry policies with exponential backoff on all external model and vector-store calls.
- **Cost Visibility:** `tiktoken`-based token/cost accounting across retrieval and generation.
- **Optional Observability:** One-line integration with Arize Phoenix for workflow tracing.
- **CI/CD:** GitHub Actions pipeline with ruff linting; architecture decisions captured in `docs/adr/`.

---

## Tech Stack

| Layer              | Technology                                                  |
| :----------------- | :---------------------------------------------------------- |
| Language           | Python 3.11+                                                |
| API Framework      | FastAPI + uvicorn                                           |
| Orchestration      | LlamaIndex Workflows (event-driven, `Context`-based steps)  |
| Vector Store       | Chroma Cloud (`chromadb>=0.6.0`) — hybrid dense+sparse, RRF |
| Semantic Cache     | Redis 7                                                     |
| Job Queue          | RQ (`src/infra/queue.py`, `src/worker.py`)                  |
| Persistence        | PostgreSQL 15 + SQLAlchemy (job/document records)           |
| Embeddings         | `BAAI/bge-small-en-v1.5` (HuggingFace)                      |
| Reranker           | `BAAI/bge-reranker-v2-m3` (FlagEmbedding)                   |
| LLM                | Groq (Llama 3.3 70B for retrieval, Llama 3.1 8B for metadata enrichment) |
| Config             | Pydantic v2 / pydantic-settings                             |
| Resilience         | tenacity (exponential backoff)                              |
| Observability      | Arize Phoenix (optional)                                    |
| Tooling            | ruff, pytest                                                 |
| Infrastructure     | Docker & Docker Compose                                     |

---

## Architecture

### Component Flow

```text
┌────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
│   CLI /    │──▶│  FastAPI      │──▶│  Ingestion      │──▶│  Chroma Cloud    │
│   API      │   │  (src/api)    │   │  Workflow       │   │  (dense+sparse)  │
└────────────┘   └───────────────┘   │  (RQ worker)    │   └────────┬─────────┘
                                     │                 │            │
                                     │  PIIMasker      │            │
                                     │  → chunking     │            │
                                     │  → LLM enrich   │            │
                                     └─────────────────┘            ▼
                                                           ┌──────────────────┐
                                                           │  Retrieval       │
                                                           │  Workflow        │
┌────────────┐   ┌───────────────┐   ┌─────────────────┐   │  (HyDE, rerank,  │
│  Query     │──▶│  Semantic     │──▶│  Relevance      │──▶│  refine loop)    │
│  (API/CLI) │   │  Cache (Redis)│   │  judgment       │   └──────────────────┘
└────────────┘   └───────────────┘   └─────────────────┘            │
                                                                    ▼
                                                          ┌──────────────────┐
                                                          │  Groq answer     │
                                                          │  (+ sources)     │
                                                          └──────────────────┘
```

### Retrieval Pipeline (LLM Query Path)

```text
query ──► semantic cache lookup (embedding similarity ≥ 0.85 → cached answer)
   │ miss
   ▼
HyDE: generate hypothetical document → embed
   ▼
hybrid vector search (dense + sparse, RRF fusion)
   ▼
relevance judgment → refine query & re-search (loop) 
   ▼
cross-encoder rerank → LongContextReorder
   ▼
Groq generation + token/cost accounting
   ▼
answer (+ cache for future identical queries)
```

### Project Structure

```text
Project-Aether/
├── main.py                          # Entry: CLI (default) or --api (uvicorn)
├── src/
│   ├── api/app.py                   # FastAPI: /health, /ingest, /jobs/{id}, /query
│   ├── config/settings.py           # Pydantic-settings, env-driven config
│   ├── core/pii.py                  # Regex PII masker (email, phone)
│   ├── pipeline/
│   │   ├── ingestion.py             # Event-driven ingestion workflow
│   │   └── retrieval.py             # Retrieval workflow (HyDE, judgment, rerank)
│   ├── services/
│   │   ├── chroma.py                # Chroma Cloud adapter (hybrid collection)
│   │   └── redis.py                 # Semantic cache (degraded-mode aware)
│   ├── infra/queue.py               # RQ queue wiring
│   ├── jobs/ingestion.py            # Background ingestion job
│   ├── db/session.py                # SQLAlchemy session
│   ├── models/db.py                 # IngestionJob ORM model
│   └── utils/                       # Logger, token counter
├── scripts/migrate_to_chroma.py     # Qdrant → Chroma Cloud migration
├── docs/adr/                        # Architecture Decision Records (001–003)
├── tests/                           # pytest suite (isolated via mocking)
├── Dockerfile
├── docker-compose.yml               # api + worker + redis + postgres
├── .env.example
└── .github/workflows/ci.yml
```

---

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Groq API Key (free tier)
- Chroma Cloud credentials (host, tenant, database, API key)

### Installation

```bash
git clone <REPOSITORY_URL>
cd Project-Aether

pip install -r requirements.txt

cp .env.example .env
# Set GROQ_API_KEY, CHROMA_HOST, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
```

### Running the Full Stack (Docker)

```bash
docker compose up -d --build
```

This brings up four services: the FastAPI API (`:8000`), an RQ ingestion worker, Redis, and PostgreSQL 15.

### Running Locally

**Ingest documents** (reads `DATA_DIR`, defaults to `./data`):

```bash
python main.py
```

**Serve the API:**

```bash
python main.py --api
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

### Example API Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this project?"}'
```

Response includes the generated answer and a `from_cache` flag indicating whether the result was served from the semantic cache.

### Running Tests

```bash
pytest tests/
```

The suite covers all layers — ingestion, retrieval, splitter, and API — with extensive mocking for full isolation.

---

## Configuration & Environment Variables

Copy `.env.example` to `.env` and adjust for your environment.

| Variable                    | Default                    | Description                                          |
| :-------------------------- | :------------------------- | :--------------------------------------------------- |
| `GROQ_API_KEY`              | *(required)*               | Groq credentials for retrieval & enrichment LLMs.    |
| `CHROMA_HOST`               | `api.trychroma.com`        | Chroma Cloud host.                                   |
| `CHROMA_API_KEY`            | *(required)*               | Chroma Cloud API key.                                |
| `CHROMA_TENANT`             | *(required)*               | Chroma Cloud tenant ID.                              |
| `CHROMA_DATABASE`           | `RAGabaoun`                | Chroma Cloud database name.                          |
| `CHROMA_COLLECTION`         | `project_aether_docs`      | Vector collection name (dense + sparse config).      |
| `REDIS_HOST`                | `localhost`                | Redis host for the semantic cache.                   |
| `REDIS_PORT`                | `6379`                     | Redis port.                                          |
| `SEMANTIC_CACHE_THRESHOLD`  | `0.85`                     | Embedding similarity threshold for cache hits.       |
| `DATABASE_URL`              | `postgresql://user:password@postgres:5432/aether` | Job persistence DSN.          |
| `LOG_LEVEL`                 | `INFO`                     | Logging verbosity.                                   |
| `DATA_DIR`                  | `./data`                   | Document source directory for ingestion.             |
| `PHOENIX_COLLECTOR_ENDPOINT`| `http://localhost:6006`    | Arize Phoenix collector endpoint (observability).    |
| `DEBUG`                     | `false`                    | When `true`, surfaces detailed error payloads.       |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | — | Legacy Qdrant settings (migration source). |

---

## API Reference

| Method | Route            | Description                                                      |
| :----- | :--------------- | :--------------------------------------------------------------- |
| `GET`  | `/health`        | Service health + retrieval-readiness probe.                     |
| `POST` | `/ingest`        | Triggers background document ingestion; returns `job_id` (202).  |
| `GET`  | `/jobs/:id`      | Polls the status of an ingestion job (`PENDING`/`...`).           |
| `POST` | `/query`         | Executes the retrieval pipeline; returns answer + `from_cache`.  |

---

## Roadmap

- **Streaming responses** over SSE for the `/query` endpoint.
- **Multi-tenant collection isolation** and per-tenant cache namespaces.
- **Chunk-level source citations** with confidence scores in API responses.
- **Incremental indexing** with change detection to avoid full re-ingestion.
- **Evaluation harness** (retrieval hit-rate / answer faithfulness) integrated into CI.
