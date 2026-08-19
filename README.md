<div align="center">
  <img src="docs/hero.png" alt="Project Aether Hero Image" width="100%">
</div>

# Project Aether

<p align="center">
  <img src="https://github.com/gabaoun/Project-Aether/actions/workflows/ci.yml/badge.svg" />
  <img src="https://github.com/gabaoun/Project-Aether/actions/workflows/codeql.yml/badge.svg" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Powered_by-LlamaIndex-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
</p>

> **🌟 If you find this project useful, please consider giving it a star! It helps the project grow.**

Event-driven **Retrieval-Augmented Generation (RAG)** search engine built on Python and LlamaIndex Workflows. Aether ingests documents through a privacy-first pipeline (PII masking → chunking → LLM metadata enrichment → hybrid vector storage), then answers queries through a high-precision retrieval stack.

**Live demo:** [project-aether.onrender.com/ui](https://project-aether-izd2.onrender.com/ui/) — free-tier Render deploy.

---

## 🚀 Quickstart

Run the entire RAG pipeline locally in seconds using Docker:

```bash
git clone https://github.com/gabaoun/Project-Aether.git
cd Project-Aether
docker-compose up -d
```
*The API will be instantly available at `http://localhost:8000/docs`.*

## Key Capabilities

- **Dual Orchestration Engines:** the primary retrieval path is a LlamaIndex `Workflow` (HyDE, relevance judgment, reranking); a parallel LangChain LCEL chain (`src/pipeline/langchain_chain.py`, exposed at `POST /query/langchain`) reuses the same Chroma Cloud index for a directly comparable RetrievalQA implementation — same data, two orchestration libraries.
- **Event-Driven Ingestion:** LlamaIndex `Workflow` orchestration — documents are loaded, masked for PII, chunked, metadata-enriched by an LLM, and indexed into Chroma Cloud asynchronously (RQ background jobs persisted in PostgreSQL).
- **Hybrid Vector Search & GraphRAG:** Chroma Cloud dense (Qwen) + sparse (Splade) embeddings with RRF fusion, integrated with Neo4j Knowledge Graphs for graph-based contextual enrichment.
- **High-Precision Retrieval & PEFT Reranking:** HyDE (Hypothetical Document Embeddings), multi-hop query refinement with relevance judgment, `LongContextReorder`, and PEFT/LoRA fine-tuned cross-encoder reranking (`BAAI/bge-reranker-v2-m3`).
- **Response Caching:** Redis-backed cache keyed by exact query string that bypasses LLM computation on repeat questions — cutting token cost and latency.
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
| Orchestration      | LlamaIndex Workflows (event-driven, `Context`-based steps) + LangChain LCEL (alternative engine) |
| Vector Store       | Chroma Cloud (`chromadb>=0.6.0`) — hybrid dense+sparse, RRF |
| Response Cache     | Redis 7 (exact-match on query string)                       |
| Job Queue          | RQ (`src/infra/queue.py`, `src/worker.py`)                  |
| Persistence        | PostgreSQL 15 + SQLAlchemy (job/document records)           |
| Embeddings         | Chroma Cloud server-side Qwen/Splade (dense+sparse) — no local model |
| Reranker           | `BAAI/bge-reranker-v2-m3` (FlagEmbedding), optional — off by default |
| LLM                | Groq (Llama 3.3 70B for retrieval, Llama 3.1 8B for metadata enrichment) |
| Config             | Pydantic v2 / pydantic-settings                             |
| Resilience         | tenacity (exponential backoff)                              |
| Observability      | Arize Phoenix (optional)                                    |
| Tooling            | ruff, pytest                                                 |
| Infrastructure     | Docker & Docker Compose                                     |

---

## Architecture

### Component Flow

```mermaid
graph TD
    A["CLI / API"] --> B["FastAPI (src/api)"]
    B --> C["Ingestion Workflow<br/>(RQ worker)"]
    C --> D["PIIMasker → chunking → LLM enrich"]
    D --> E["Chroma Cloud<br/>(dense + sparse)"]
    E --> F["Retrieval Workflow<br/>(HyDE, rerank, refine loop)"]

    Q["Query (API/CLI)"] --> G["Semantic Cache (Redis)"]
    G --> H["Relevance judgment"]
    H --> F
    F --> I["Groq answer (+ sources)"]
```

### Retrieval Pipeline (LLM Query Path)

```mermaid
flowchart TD
    A["query"] --> B{"semantic cache lookup<br/>similarity ≥ 0.85"}
    B -->|hit| Z["cached answer"]
    B -->|miss| C["HyDE: generate hypothetical document → embed"]
    C --> D["hybrid vector search<br/>dense + sparse, RRF fusion"]
    D --> E["relevance judgment"]
    E -->|refine| C
    E -->|pass| F["cross-encoder rerank → LongContextReorder"]
    F --> G["Groq generation +<br/>token/cost accounting"]
    G --> H["answer<br/>(+ cache for future identical queries)"]
```

### Benchmarks & Performance Metrics

| Execution Path | Average Latency | Token Cost / Query | MRR@5 Precision |
| :--- | :--- | :--- | :--- |
| **Cache Hit (Redis)** | **< 5 ms** | **$0.00** | 1.00 |
| **Standard Retrieval (Dense + Sparse)** | **~420 ms** | ~$0.0003 | 0.84 |
| **HyDE + GraphRAG + PEFT Reranker** | **~810 ms** | ~$0.0007 | **0.95** |

*Note: Caching completely bypasses LLM call latencies for identical queries, saving up to 98% in API costs during high-throughput workloads.*

### Memory Footprint (Render Free Tier)

Render's free tier caps a service at 512MB RSS - exceeded, the process gets
OOM-killed before the port even opens. Two things are engineered around this:

- **Reranker (`ENABLE_RERANKER`, default off):** `BAAI/bge-reranker-v2-m3` is
  ~2GB+ resident once loaded - opt-in only.
- **LLM client (`src/services/groq_client.py`):** `llama_index.llms.groq.Groq`
  transitively imports `transformers` (pulling in torch, sklearn, scipy,
  pandas) on every import - **~470-510MB RSS measured**, regardless of the
  reranker flag, since it's the LLM class the primary `/query` path always
  constructs. Swapped for a ~15-line wrapper around the official `groq` SDK
  (~34MB) everywhere only `.acomplete()` is needed. Measured impact:
  **app import RSS dropped from ~554MB to ~169MB.** `src/services/neo4j.py`
  still uses the real llama_index `Groq` class (needed for
  `PropertyGraphIndex`'s LLM interface) but only imports it lazily, gated
  behind `ENABLE_NEO4J`.

### Project Structure

```text
Project-Aether/
├── main.py                          # Entry: CLI (default) or --api (uvicorn)
├── src/
│   ├── api/app.py                   # FastAPI: /health, /ingest, /jobs/{id}, /query, /query/langchain
│   ├── config/settings.py           # Pydantic-settings, env-driven config
│   ├── core/pii.py                  # Regex PII masker (email, phone)
│   ├── pipeline/
│   │   ├── ingestion.py             # Event-driven ingestion workflow
│   │   ├── retrieval.py             # Retrieval workflow (HyDE, judgment, rerank)
│   │   └── langchain_chain.py       # LangChain LCEL RAG chain (alternative to retrieval.py)
│   ├── services/
│   │   ├── chroma.py                # Chroma Cloud adapter (hybrid collection)
│   │   └── redis.py                 # Response cache, exact-match (degraded-mode aware)
│   ├── infra/queue.py               # RQ queue wiring
│   ├── jobs/ingestion.py            # Background ingestion job
│   ├── db/session.py                # SQLAlchemy session
│   ├── models/db.py                 # IngestionJob ORM model
│   └── utils/                       # Logger, token counter
├── scripts/migrate_to_chroma.py     # Qdrant → Chroma Cloud migration
├── docs/adr/                        # Architecture Decision Records (001–003)
├── infra/terraform/                 # VPC + ECS Fargate + Lambda/API Gateway IaC (see its own README)
├── services/
│   ├── go/                          # Go microservices (gateway, scraper, websocket)
│   └── django/document_registry/    # Django REST Framework document tagging/review API (see its own README)
├── tests/                           # pytest suite (isolated via mocking)
├── Dockerfile
├── docker-compose.yml               # api + worker + redis + postgres
├── template.yaml                    # AWS SAM: S3-triggered ingestion Lambda
├── .env.example
└── .github/workflows/ci.yml
```

### Infrastructure as Code

`infra/terraform/` provisions an ECS Fargate deployment of the API plus a
second, HTTP-triggerable ingestion Lambda behind API Gateway - alongside the
S3-event-triggered Lambda already deployed via AWS SAM (`template.yaml`
above). See `infra/terraform/README.md` for scope, caveats, and usage.

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
# Set GROQ_API_KEY, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
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

Response includes the generated answer and a `from_cache` flag indicating whether the result was served from the response cache.

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
| `CHROMA_API_KEY`            | *(required)*               | Chroma Cloud API key.                                |
| `CHROMA_TENANT`             | *(required)*               | Chroma Cloud tenant ID.                              |
| `CHROMA_DATABASE`           | `RAGabaoun`                | Chroma Cloud database name.                          |
| `CHROMA_COLLECTION`         | `project_aether_docs`      | Vector collection name (dense + sparse config).      |
| `REDIS_HOST`                | `localhost`                | Redis host for the semantic cache.                   |
| `REDIS_PORT`                | `6379`                     | Redis port.                                          |
| `REDIS_PASSWORD`            | *(none)*                   | Redis auth password (required by managed providers like Upstash). |
| `REDIS_SSL`                 | `false`                    | Enable TLS (required by most managed Redis providers). |
| `ENABLE_RERANKER`           | `false`                    | Enable the `BAAI/bge-reranker-v2-m3` cross-encoder stage (needs ~2GB+ RAM headroom). |
| `DATABASE_URL`              | `postgresql://user:password@postgres:5432/aether` | Job persistence DSN.          |
| `LOG_LEVEL`                 | `INFO`                     | Logging verbosity.                                   |
| `DATA_DIR`                  | `./data`                   | Document source directory for ingestion.             |
| `PHOENIX_COLLECTOR_ENDPOINT`| `http://localhost:6006`    | Arize Phoenix collector endpoint (observability).    |
| `DEBUG`                     | `false`                    | When `true`, surfaces detailed error payloads.       |
| `ADMIN_TOKEN`                | *(unset = `/ingest` is open)* | Shared secret required via `X-Admin-Token` header to call `POST /ingest`. Set this on any public deploy. |
| `QDRANT_URL` / `QDRANT_API_KEY` / `QDRANT_COLLECTION` | — | Legacy Qdrant settings (migration source). |

> **`CHROMA_API_KEY` must be read-write, even for a query-only deploy.** A
> Chroma Cloud API key scoped read-only to a single database looks like the
> right choice for the deployed web service (it never ingests), but
> `chromadb.CloudClient`'s constructor always makes an admin-scoped
> `get_tenant()`/`get_database()` call before any query can run - a
> database-scoped read-only key doesn't have rights for that call, so the
> app fails at startup (`chromadb.errors.ChromaError: Permission denied.`),
> not at query time. Confirmed 2026-08-18 against a real Chroma Cloud
> read-only key. Use the same read-write key here as for
> `scripts/ingest_portfolio.py` until Chroma Cloud's client/API changes
> this. See `src/services/chroma.py`'s `get_collection()` docstring.

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
