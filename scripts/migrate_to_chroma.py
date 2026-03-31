import asyncio
from qdrant_client import QdrantClient
from src.services.chroma import ChromaService
from src.config.settings import settings
from src.utils.logger import logger

async def migrate():
    logger.info("Starting migration from Qdrant to Chroma Cloud...")
    
    # 1. Initialize Qdrant Client
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    # 2. Initialize Chroma Service
    chroma_service = ChromaService()
    
    # 3. Fetch points from Qdrant
    try:
        # We need to scroll through all points
        points, next_page_offset = qdrant_client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            with_payload=True,
            with_vectors=False # We let Chroma Cloud re-embed or we could try to migration vectors if models match
        )
        
        while True:
            docs_to_migrate = []
            for point in points:
                # Extract text and metadata from Qdrant payload
                # Note: LlamaIndex Qdrant payload usually has '_node_content' or 'text'
                payload = point.payload
                text = payload.get('text') or payload.get('_node_content')
                
                if not text:
                    logger.warning(f"Point {point.id} has no text content. Skipping.")
                    continue
                
                docs_to_migrate.append({
                    "id": str(point.id),
                    "text": text,
                    "metadata": payload
                })
            
            if docs_to_migrate:
                await chroma_service.upsert_documents(docs_to_migrate)
                logger.info(f"Migrated {len(docs_to_migrate)} documents.")
            
            if not next_page_offset:
                break
                
            points, next_page_offset = qdrant_client.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
        logger.info("Migration complete!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")

if __name__ == "__main__":
    asyncio.run(migrate())
