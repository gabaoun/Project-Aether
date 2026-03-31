import os
import asyncio
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

async def main():
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collection_name = "project_aether_docs"
    
    ingestion_wf = IngestionWorkflow(qdrant_client=qdrant_client, collection_name=collection_name)
    
    data_dir = settings.data_dir
    if os.path.exists(data_dir) and os.listdir(data_dir):
        logger.info(f"Starting ingestion from {data_dir}...")
        index = await ingestion_wf.run(input_dir=data_dir)
    else:
        logger.error(f"Data directory '{data_dir}' is empty or missing.")
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

if __name__ == "__main__":
    asyncio.run(main())
