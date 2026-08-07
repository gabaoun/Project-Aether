import asyncio
import os
import sys
from pathlib import Path

# Add project root to Python path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.ingestion import IngestionWorkflow, StartEvent
from src.utils.logger import logger
from src.config.settings import settings


async def main():
    logger.info("Starting Portfolio Ingestion Workflow...")
    
    portfolio_dir = Path(settings.data_dir) / "portfolio"
    
    if not portfolio_dir.exists():
        logger.error(f"Portfolio directory not found: {portfolio_dir}")
        logger.error("Please create it and add your candidate profile documents.")
        sys.exit(1)
        
    workflow = IngestionWorkflow(timeout=300)
    
    try:
        # Pass input_dir to the StartEvent for load_documents
        result = await workflow.run(input_dir=str(portfolio_dir))
        logger.info("Ingestion complete. Portfolio documents are now in Chroma Cloud.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
