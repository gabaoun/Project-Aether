"""
Project Aether - Entry Point
Author: Gabriel (Gabaoun) Penha
"""
import os
import asyncio
import logging
from qdrant_client import QdrantClient
from src.processing.ingestion_workflow import IngestionWorkflow
from src.retrieval.retrieval_workflow import RetrievalWorkflow, StreamingStatusEvent
from src.utils.config import settings
from llama_index.core import set_global_handler

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level))

# Setup observability
try:
    set_global_handler("arize_phoenix", endpoint=settings.phoenix_collector_endpoint)
    print(f"Observability enabled via Arize Phoenix at {settings.phoenix_collector_endpoint}")
except ImportError:
    print("Arize Phoenix not installed. Skipping observability setup.")

async def main():
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    collection_name = "project_aether_docs"
    
    ingestion_wf = IngestionWorkflow(qdrant_client=qdrant_client, collection_name=collection_name)
    
    data_dir = settings.data_dir
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"Starting ingestion for documents in {data_dir}...")
        index = await ingestion_wf.run(input_dir=data_dir)
    else:
        print("Data directory is empty or does not exist. Skipping ingestion.")
        return

    retrieval_wf = RetrievalWorkflow(index=index)
    
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        print(f"\nProcessing query: {query}")
        
        # Stream workflow events
        handler = retrieval_wf.run(query=query)
        
        async for event in handler.stream_events():
            if isinstance(event, StreamingStatusEvent):
                print(f"[Streaming] ⏳ {event.status}")
                
        result = await handler
        
        print("\n--- Answer ---")
        if result.get("from_cache"):
            print("[✅ RETURNED FROM REDIS CACHE]")
        print(result["answer"])
        print("\n--- Sources ---")
        for node in result.get("source_nodes", []):
            file_name = node.metadata.get('file_name', 'Unknown file')
            print(f"- {file_name}")

if __name__ == "__main__":
    asyncio.run(main())