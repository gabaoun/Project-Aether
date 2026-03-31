from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from src.pipeline.ingestion import IngestionWorkflow
from src.pipeline.retrieval import RetrievalWorkflow
from src.config.settings import settings
from src.utils.logger import logger
import os

app = FastAPI(title="Project Aether RAG API")

# Global variables for index and workflow
index = None
retrieval_wf = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    from_cache: bool

@app.on_event("startup")
async def startup_event():
    global index, retrieval_wf
    
    try:
        qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        
        ingestion_wf = IngestionWorkflow(
            qdrant_client=qdrant_client, 
            collection_name=settings.qdrant_collection
        )
        
        if os.path.exists(settings.data_dir) and os.listdir(settings.data_dir):
            logger.info(f"Initializing ingestion from {settings.data_dir}...")
            index = await ingestion_wf.run(input_dir=settings.data_dir)
            retrieval_wf = RetrievalWorkflow(index=index)
            logger.info("API Startup: Ingestion complete and index ready.")
        else:
            logger.warning(f"Data directory '{settings.data_dir}' is empty or missing. API in degraded mode.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.warning("API starting in degraded mode due to infrastructure or ingestion error.")

@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": index is not None}

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
