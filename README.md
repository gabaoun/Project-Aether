# Project Aether: Event-Driven RAG Engine

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" />
</p>

## 📌 Overview
Project Aether is a high-performance **Retrieval-Augmented Generation (RAG)** pipeline and search engine built with Python, LlamaIndex, and FastAPI. It orchestrates asynchronous data ingestion workflows, semantic caching, and a robust data compliance layer to ensure enterprise-ready LLM inference.

## ⚙️ Core Engineering Features
- **Asynchronous Data Pipelines:** Orchestrates event-driven ingestion workflows to chunk, transform (Pandas/NumPy), and index unstructured data.
- **Data Compliance (PII Masking):** A dedicated privacy-centric layer designed to automatically detect and sanitize sensitive user information via regex before vectorization.
- **Advanced Retrieval Architecture:** Implements HyDE (Hypothetical Document Embeddings), query refinement loops, and relevance judgment steps.
- **Semantic Caching (Redis HNSW):** Utilizes a Redis layer to perform rapid semantic lookups, drastically reducing latency and optimizing LLM API token consumption.
- **Resilient Infrastructure:** Features a degraded mode that automatically detects connection failures to external vector stores (Qdrant), gracefully falling back to prevent system crashes.

## 🚀 Running the API
The backend exposes REST endpoints via FastAPI.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
python main.py --api
```

*Access the Swagger UI at http://localhost:8000/docs to test the RAG endpoints.*
