# Project Aether: RAG Pipeline with Event-Driven Workflows

## Overview
Project Aether is a high-performance Retrieval-Augmented Generation (RAG) search engine built with Python and LlamaIndex. It features an event-driven orchestration pipeline (Workflows), semantic caching, a data compliance layer, and interfaces via both a CLI and a FastAPI service.

## Features
- **FastAPI Interface:** A RESTful API layer providing low-latency endpoints to interact with the RAG engine.
- **Event-Driven Ingestion:** Orchestrates asynchronous ingestion workflows to process and index documents.
- **Advanced Retrieval:** Implements HyDE (Hypothetical Document Embeddings), query refinement loops, and relevance judgment steps.
- **Semantic Caching:** Utilizes a Redis caching layer (with HNSW index configurations) to perform fast semantic lookups and optimize LLM token consumption.
- **PII Masking Layer:** A dedicated privacy-centric layer designed to automatically detect and mask sensitive user information (PII) before ingestion.
- **Resilient Degraded Mode:** Automatically detects connection failures to external vector stores or caches, adjusting behavior gracefully to prevent application crashes.
- **Fault Tolerance:** Built-in retry mechanisms using exponential backoff to handle transient API issues.

## Run CLI
To start the interactive CLI mode:
```bash
python main.py
```

## Run API
To start the FastAPI server:
```bash
python main.py --api
```
The API will be available at `http://localhost:8000`. You can access the automatic documentation (Swagger UI) at `http://localhost:8000/docs`.

## Example API Request
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this project?"}'
```

## Tech Stack
- **Language:** Python 3.11+
- **API Framework:** FastAPI
- **Orchestration:** LlamaIndex (Workflows)
- **Vector Database:** Qdrant
- **Cache:** Redis
- **LLM:** OpenAI (GPT-4o, GPT-4o-mini)
- **Configuration:** Pydantic Settings

## Key Technical Points
- **Modular Refactoring:** Logic is split into `core`, `services`, `pipeline`, `models`, `config`, `api`, and `utils`.
- **Mocked Testing:** The project includes unit tests for all layers (ingestion, retrieval, splitter, and API) with extensive mocking to ensure tests run in isolation.
- **Degraded Mode Infrastructure:** The system detects connection failures to Redis or Qdrant and adjusts its behavior (e.g., skipping cache) instead of crashing.

## Core Configurations
- **Regex-based PII Masker:** Employs high-speed regular expression processing to sanitize email addresses and phone numbers.
- **Semantic Caching Layer:** Leverages Redis key-value stores to bypass heavy LLM computation for previously processed semantic queries.

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- OpenAI API Key

### Installation
1. Clone the repository and navigate to the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API Key and other settings
   ```
4. Start infrastructure (Optional for basic runs):
   ```bash
   docker-compose up -d
   ```

### Testing
Run the test suite using pytest:
```bash
pytest tests/
```
