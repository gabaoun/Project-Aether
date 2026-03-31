# Project Aether: RAG Pipeline with Event-Driven Workflows

## Overview
Project Aether is a Retrieval-Augmented Generation (RAG) system built with Python and LlamaIndex. It implements a document ingestion and retrieval pipeline using an event-driven architecture (Workflows) to handle complex tasks like query transformation, metadata enrichment, and semantic caching.

The project is designed as a modular reference for building RAG applications that require more than simple linear processing, incorporating retries, asynchronous operations, and a clear separation of concerns.

## Features
- **Event-Driven Ingestion:** Processes documents through a series of discrete steps (Loading -> PII Masking -> Semantic Splitting -> Enrichment -> Indexing).
- **Advanced Retrieval:** Implements HyDE (Hypothetical Document Embeddings), query refinement loops, and relevance judgment (Chain-of-Thought) before generating answers.
- **Semantic Caching:** Uses Redis to store and retrieve previously generated answers for identical or highly similar queries to reduce LLM latency and cost.
- **PII Masking:** Basic regex-based masking of emails and phone numbers during the ingestion phase.
- **Resiliency:** Uses `tenacity` for exponential backoff retries on LLM and database operations.
- **Memory Efficiency:** Uses Python generators during document splitting to handle larger datasets without high memory consumption.

## Tech Stack
- **Language:** Python 3.11+
- **Orchestration:** LlamaIndex (Workflows)
- **Vector Database:** Qdrant
- **Cache:** Redis
- **LLM:** OpenAI (GPT-4o, GPT-4o-mini)
- **Embeddings:** HuggingFace (BGE models)
- **Configuration:** Pydantic Settings

## Key Technical Points
- **Modular Refactoring:** Logic is split into `core` (business logic), `services` (external integrations), `pipeline` (workflow orchestration), and `models` (data structures).
- **Asynchronous Execution:** Heavy use of `asyncio` for non-blocking I/O, particularly in PII masking and LLM calls.
- **Custom Splitter:** Implements a `SemanticDoubleMergingSplitter` which performs an initial semantic split and then merges small chunks that fall below a minimum size threshold.

## Design Decisions
- **LlamaIndex Workflows over Pipelines:** Chosen to allow for non-linear logic, such as the query refinement loop in the retrieval workflow which can re-run if initial results are deemed irrelevant.
- **BGE-Reranker:** Integrated to improve precision by re-evaluating the top retrieved nodes using a cross-encoder model.
- **Strict Typing:** All major functions and classes use Python type hints for better maintainability and error detection.

## Limitations
- **Regex-based PII:** The current PII masker uses basic regular expressions and is not a substitute for a production-grade NER (Named Entity Recognition) system.
- **Simplified Semantic Cache:** The current implementation uses exact string matching in Redis for the cache keys rather than true vector-based similarity search.
- **Single Collection:** Currently hardcoded to use a single Qdrant collection for all documents.

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- OpenAI API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Project-Aether.git
   cd Project-Aether
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Setup environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API Key and other settings
   ```
4. Start infrastructure:
   ```bash
   docker-compose up -d
   ```

### Usage
Ensure you have documents in the `./data` directory (as specified in your `.env`), then run:
```bash
python main.py
```

## Testing
Run the test suite using pytest:
```bash
pytest tests/
```

## Purpose
This project was developed to demonstrate a technically sound approach to building RAG systems. It focuses on clean architecture, error handling, and implementing advanced RAG patterns in a way that is maintainable and extensible.
