import os
import asyncio
import argparse
import uvicorn
from qdrant_client import QdrantClient
from src.pipeline.ingestion import IngestionWorkflow
from src.pipeline.retrieval import RetrievalWorkflow, StreamingStatusEvent
from src.config.settings import settings
from src.utils.logger import logger
from llama_index.core import set_global_handler

# Optional Observability
try:
    set_global_handler("arize_phoenix", endpoint=settings.phoenix_collector_endpoint)
    logger.info(f"Observability enabled at {settings.phoenix_collector_endpoint}")
except ImportError:
    logger.warning("Arize Phoenix not installed. Skipping observability.")

async def run_cli():
    """Runs the project in interactive CLI mode."""
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    ingestion_wf = IngestionWorkflow(
        qdrant_client=qdrant_client, 
        collection_name=settings.qdrant_collection
    )
    
    data_dir = settings.data_dir
    if os.path.exists(data_dir) and os.listdir(data_dir):
        logger.info(f"Starting ingestion from {data_dir}...")
        index = await ingestion_wf.run(input_dir=data_dir)
    else:
        logger.error(f"Data directory '{data_dir}' is empty or missing. Add documents and restart.")
        return

    retrieval_wf = RetrievalWorkflow(index=index)
    
    while True:
        try:
            query = input("\nEnter query (or 'exit'): ")
            if query.lower() == 'exit':
                break
                
            handler = retrieval_wf.run(query=query)
            
            async for event in handler.stream_events():
                if isinstance(event, StreamingStatusEvent):
                    print(f"⏳ {event.status}")
                    
            result = await handler
            
            print("\n--- Answer ---")
            if result.get("from_cache"):
                print("[CACHED]")
            print(result["answer"])
            print("\n--- Sources ---")
            for node in result.get("source_nodes", []):
                print(f"- {node.metadata.get('file_name', 'Unknown')}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

def run_api():
    """Runs the project as a FastAPI server."""
    logger.info("Starting Project Aether API...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project Aether: RAG Engine")
    parser.add_argument("--api", action="store_true", help="Run in API mode")
    args = parser.parse_args()
    
    if args.api:
        run_api()
    else:
        asyncio.run(run_cli())
