import asyncio

from src.config.settings import settings
from src.db.session import SessionLocal
from src.models.db import IngestionJob
from src.pipeline.ingestion import IngestionWorkflow
from src.utils.logger import logger


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

        # workflow.run() calls asyncio.create_task() internally, so it must be
        # awaited from inside an already-running loop - passing it directly to
        # asyncio.run() evaluates it before that loop exists and raises
        # "RuntimeError: no running event loop". Wrapping it in a coroutine
        # that awaits it from inside asyncio.run() fixes this.
        async def _run_workflow() -> None:
            await workflow.run(input_dir=settings.data_dir)

        asyncio.run(_run_workflow())

        job.status = "COMPLETED"
        logger.info(f"Ingestion job {job_id} completed successfully.")
    except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
        logger.error(f"Ingestion job {job_id} failed: {e}")
        job.status = "FAILED"
    finally:
        db.commit()
        db.close()
