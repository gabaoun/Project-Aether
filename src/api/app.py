from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.chroma import ChromaService
from src.pipeline.ingestion import IngestionWorkflow
from src.pipeline.retrieval import RetrievalWorkflow
from src.config.settings import settings
from src.utils.logger import logger
import os

app = FastAPI(title="Project Aether RAG API")

# Global variables for chroma service and workflow
chroma_service = None
retrieval_wf = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    from_cache: bool

@app.on_event("startup")
async def startup_event():
    global chroma_service, retrieval_wf
    
    try:
        ingestion_wf = IngestionWorkflow()
        
        if os.path.exists(settings.data_dir) and os.listdir(settings.data_dir):
            logger.info(f"Initializing ingestion from {settings.data_dir}...")
            # ingestion_wf.run returns chroma_service (StopEvent result)
            chroma_service = await ingestion_wf.run(input_dir=settings.data_dir)
            retrieval_wf = RetrievalWorkflow(chroma_service=chroma_service)
            logger.info("API Startup: Ingestion complete and Chroma Cloud index ready.")
        else:
            logger.warning(f"Data directory '{settings.data_dir}' is empty or missing. API in degraded mode.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.warning("API starting in degraded mode due to infrastructure or ingestion error.")

@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": chroma_service is not None}

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
        raise HTTPException(status_code=500, detail="An error occurred during query processing.")
