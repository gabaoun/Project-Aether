from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from src.services.chroma import ChromaService
from src.pipeline.retrieval import RetrievalWorkflow
from src.config.settings import settings
from src.utils.logger import logger
from src.db.session import get_db
from src.models.db import IngestionJob
from src.infra.queue import get_queue
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

# Global variables for chroma service and workflow
chroma_service = None
retrieval_wf = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_service, retrieval_wf
    
    try:
        # Initialize search infrastructure on startup
        chroma_service = ChromaService()
        retrieval_wf = RetrievalWorkflow(chroma_service=chroma_service)
        logger.info("API Startup: Chroma Cloud retrieval ready.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.warning("API starting in degraded mode due to infrastructure error.")
    
    yield

app = FastAPI(title="Project Aether RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    from_cache: bool

class IngestResponse(BaseModel):
    job_id: str

class JobStatusResponse(BaseModel):
    id: str
    status: str

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
    except Exception as e:
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
    
    return JobStatusResponse(id=str(job.id), status=job.status)

@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    global retrieval_wf
    
    if not retrieval_wf:
        raise HTTPException(status_code=503, detail="Search index is not initialized.")
    
    try:
        result = await retrieval_wf.run(query=request.query)
        return QueryResponse(
            answer=result["answer"],
            from_cache=result.get("from_cache", False)
        )
    except Exception as e:
        logger.error(f"Query failed: {request.query} - Error: {e}")
        detail = str(e) if settings.debug else "An error occurred during query processing."
        raise HTTPException(status_code=500, detail=detail)
