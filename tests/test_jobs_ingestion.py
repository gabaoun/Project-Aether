import asyncio

import pytest

from src.jobs.ingestion import process_ingestion


class _FakeWorkflowHandler:
    """Mimics llama-index's real WorkflowHandler: touches the running event
    loop eagerly at call time (via get_running_loop), not at await time.
    This is exactly the behavior that made asyncio.run(workflow.run(...))
    crash with "no running event loop" - calling workflow.run() outside an
    already-running loop must fail the same way here.
    """
    def __init__(self):
        asyncio.get_running_loop()

    def __await__(self):
        async def _coro():
            return {"answer": "ok"}
        return _coro().__await__()


class _FakeWorkflow:
    def run(self, **kwargs):
        return _FakeWorkflowHandler()


@pytest.fixture(autouse=True)
def _mock_ingestion_workflow(mocker):
    mocker.patch('src.jobs.ingestion.IngestionWorkflow', return_value=_FakeWorkflow())


def _mock_db(mocker, job):
    mock_session = mocker.MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = job
    mocker.patch('src.jobs.ingestion.SessionLocal', return_value=mock_session)
    return mock_session


def test_process_ingestion_completes_without_a_running_event_loop(mocker):
    # process_ingestion is called synchronously by the RQ worker - no event
    # loop is running yet when this function starts. If it evaluates
    # workflow.run(...) before establishing its own loop (the original bug),
    # this raises RuntimeError immediately.
    job = mocker.MagicMock(status="PENDING")
    _mock_db(mocker, job)

    process_ingestion("job-123")

    assert job.status == "COMPLETED"


def test_process_ingestion_marks_job_failed_on_workflow_error(mocker):
    mocker.patch(
        'src.jobs.ingestion.IngestionWorkflow',
        return_value=mocker.MagicMock(run=mocker.MagicMock(side_effect=RuntimeError("boom"))),
    )
    job = mocker.MagicMock(status="PENDING")
    _mock_db(mocker, job)

    process_ingestion("job-123")

    assert job.status == "FAILED"


def test_process_ingestion_handles_missing_job(mocker):
    _mock_db(mocker, None)

    # Should return quietly, not raise, when the job id doesn't exist.
    process_ingestion("does-not-exist")
