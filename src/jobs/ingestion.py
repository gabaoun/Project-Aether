import asyncio
from src.db.session import SessionLocal
from src.models.db import IngestionJob
from src.pipeline.ingestion import IngestionWorkflow
from src.utils.logger import logger
from src.config.settings import settings

def process_ingestion(job_id: str):
    """
    Background job to process document ingestion.
    """
    db = SessionLocal()
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    
    if not job:
        logger.error(f"Job {job_id} not found in database.")
        db.close()
        return

    job.status = "PROCESSING"
    db.commit()

    try:
        logger.info(f"Starting ingestion workflow for job {job_id}")
        workflow = IngestionWorkflow()
        
        # Run the workflow
        # Note: workflow.run is async, so we need to run it in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # This shouldn't happen in a standard RQ worker but just in case
            future = asyncio.ensure_future(workflow.run(input_dir=settings.data_dir))
            loop.run_until_complete(future)
        else:
            asyncio.run(workflow.run(input_dir=settings.data_dir))

        job.status = "COMPLETED"
        logger.info(f"Ingestion job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}")
        job.status = "FAILED"
    finally:
        db.commit()
        db.close()
