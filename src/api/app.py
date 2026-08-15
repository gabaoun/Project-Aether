import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config.settings import settings
from src.db.session import get_db
from src.infra.queue import get_queue
from src.models.db import IngestionJob
from src.pipeline.retrieval import RetrievalWorkflow
from src.services.chroma import ChromaService
from src.utils.logger import logger

# Global variables for chroma service and workflow
chroma_service = None
retrieval_wf = None
langchain_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_service, retrieval_wf, langchain_chain

    try:
        # Initialize search infrastructure on startup
        chroma_service = ChromaService()
        retrieval_wf = RetrievalWorkflow(chroma_service=chroma_service)

        # langchain-core alone adds ~230-290MB RSS on import (measured
        # 2026-08-14) - same OOM risk on Render's 512MB free tier as the
        # reranker below, so it gets the same opt-in treatment: off by
        # default, and the import itself (not just the instantiation)
        # stays inside this branch so it never touches memory unless asked.
        if settings.enable_langchain_engine:
            from src.pipeline.langchain_chain import LangChainRAGChain

            langchain_chain = LangChainRAGChain(chroma_service=chroma_service)
            logger.info("LangChain engine enabled.")
        else:
            logger.info("LangChain engine disabled via settings.enable_langchain_engine (default) - skipping import to save memory on constrained hosts.")

        logger.info("API Startup: Chroma Cloud retrieval ready.")
    except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
        logger.error(f"Startup failed: {e}")
        logger.warning("API starting in degraded mode due to infrastructure error.")
    
    yield

app = FastAPI(title="Project Aether RAG API", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    from_cache: bool
    source_nodes: list[str] = []

class IngestResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    id: str
    status: str

@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Project Aether API is running", "docs": "/docs", "ui": "/ui"}

@app.get("/health")
async def health():
    return {"status": "ok", "retrieval_ready": retrieval_wf is not None}

@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_docs(db: Session = Depends(get_db)):
    """
    Trigger document ingestion as a background job.
    """
    try:
        # Create a job record in Postgres
        job = IngestionJob(status="PENDING")
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Enqueue the background task
        queue = get_queue()
        queue.enqueue("src.jobs.ingestion.process_ingestion", str(job.id))
        
        logger.info(f"Ingestion job {job.id} enqueued.")
        return IngestResponse(job_id=str(job.id))
    except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
        logger.error(f"Failed to enqueue ingestion job: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger ingestion.")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Check the status of an ingestion job.
    """
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    
    return JobStatusResponse(id=str(job.id), status=job.status or "UNKNOWN")

@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    if not retrieval_wf:
        raise HTTPException(status_code=503, detail="Search index is not initialized.")
    
    try:
        # RetrievalWorkflow's final step always returns StopEvent(result={...}) -
        # a dict - but Workflow.run()'s generic RunResultT is unbound in the stub.
        result = cast(dict[str, Any], await retrieval_wf.run(query=request.query))
        source_nodes = result.get("source_nodes", [])
        sources = []
        for n in source_nodes:
            meta = getattr(n, "metadata", {}) or {}
            fname = meta.get("file_name") or meta.get("file_path") or meta.get("filename") or meta.get("source")
            if fname:
                name = os.path.basename(str(fname))
                if name and name.lower() != "unknown" and name not in sources:
                    sources.append(name)

        return QueryResponse(
            answer=result["answer"],
            from_cache=result.get("from_cache", False),
            source_nodes=sources
        )
    except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
        logger.error(f"Query failed: {request.query} - Error: {e}")
        detail = str(e) if settings.debug else "An error occurred during query processing."
        raise HTTPException(status_code=500, detail=detail)

@app.post("/query/langchain", response_model=QueryResponse)
async def query_docs_langchain(request: QueryRequest):
    """
    Same query surface as /query, but orchestrated through a LangChain LCEL
    chain instead of the LlamaIndex Workflow - both read the same Chroma
    Cloud index, so results are directly comparable between engines.
    """
    if not langchain_chain:
        raise HTTPException(status_code=503, detail="Search index is not initialized.")

    try:
        result = await langchain_chain.aquery(request.query)
        source_nodes = []
        for meta in result.get("source_nodes", []):
            fname = meta.get("file_name") or meta.get("file_path") or meta.get("filename") or meta.get("source")
            if fname:
                name = os.path.basename(str(fname))
                if name and name.lower() != "unknown" and name not in source_nodes:
                    source_nodes.append(name)

        return QueryResponse(
            answer=result["answer"],
            from_cache=result.get("from_cache", False),
            source_nodes=source_nodes
        )
    except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
        logger.error(f"LangChain query failed: {request.query} - Error: {e}")
        detail = str(e) if settings.debug else "An error occurred during query processing."
        raise HTTPException(status_code=500, detail=detail)
