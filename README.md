# Project Aether: RAG Pipeline with Event-Driven Workflows

## Overview
Project Aether is a Retrieval-Augmented Generation (RAG) system built with Python and LlamaIndex. It implements a document ingestion and retrieval pipeline using an event-driven architecture (Workflows) and provides both a CLI and a FastAPI interface.

## Features
- **FastAPI Layer:** A RESTful API to interact with the RAG engine.
- **Event-Driven Ingestion:** Processes documents through a series of discrete steps.
- **Advanced Retrieval:** Implements HyDE, query refinement loops, and relevance judgment.
- **Semantic Caching:** Uses Redis to store and retrieve previously generated answers.
- **Degraded Mode:** Gracefully handles missing Redis/Qdrant by logging warnings and continuing if possible.
- **Resiliency:** Uses `tenacity` for exponential backoff retries.

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

## Limitations
- **Regex-based PII:** Basic regular expression masking for emails and phone numbers.
- **Simplified Cache:** Uses exact query string matching for Redis keys.

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
