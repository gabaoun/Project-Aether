import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
import uuid

# Mock dependencies before importing app
import src.infra.queue
import src.db.session

# We mock the connection and session at the module level or within fixtures
from src.api.app import app

client = TestClient(app)

@pytest.fixture
def mock_db(mocker):
    mock_session = MagicMock()
    mocker.patch("src.db.session.SessionLocal", return_value=mock_session)
    # Mock Depends(get_db)
    app.dependency_overrides[src.db.session.get_db] = lambda: mock_session
    yield mock_session
    app.dependency_overrides.clear()

@pytest.fixture
def mock_queue(mocker):
    mock_q = MagicMock()
    mocker.patch("src.api.app.get_queue", return_value=mock_q)
    return mock_q

@pytest.fixture
def mock_retrieval_wf(mocker):
    mock_wf = MagicMock()
    # Mock the run method of the workflow
    mock_wf.run = AsyncMock(return_value={
        "answer": "Test Answer",
        "from_cache": False,
        "source_nodes": []
    })
    return mock_wf

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_endpoint_uninitialized():
    # Set global retrieval_wf to None for this test
    import src.api.app as api_app
    api_app.retrieval_wf = None
    
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"]

def test_query_endpoint_success(mock_retrieval_wf, mocker):
    # Set the global retrieval_wf in the app module
    mocker.patch("src.api.app.retrieval_wf", mock_retrieval_wf)
    
    response = client.post("/query", json={"query": "What is Aether?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test Answer"
    assert data["from_cache"] is False
    
    mock_retrieval_wf.run.assert_called_once_with(query="What is Aether?")

def test_query_endpoint_error(mock_retrieval_wf, mocker):
    # Mock an error during retrieval
    mock_retrieval_wf.run = AsyncMock(side_effect=Exception("Retrieval failed"))
    mocker.patch("src.api.app.retrieval_wf", mock_retrieval_wf)
    mocker.patch("src.api.app.settings.debug", True)
    
    response = client.post("/query", json={"query": "fail me"})
    
    assert response.status_code == 500
    assert "Retrieval failed" in response.json()["detail"]

def test_ingest_endpoint(mock_db, mock_queue):
    # Mock job creation
    job_id = uuid.uuid4()
    mock_db.add.side_effect = lambda job: setattr(job, 'id', job_id)
    
    response = client.post("/ingest")
    
    assert response.status_code == 202
    assert response.json()["job_id"] == str(job_id)
    mock_queue.enqueue.assert_called_once()

def test_get_job_status(mock_db):
    job_id = str(uuid.uuid4())
    mock_job = MagicMock()
    mock_job.id = job_id
    mock_job.status = "COMPLETED"
    
    mock_db.query.return_value.filter.return_value.first.return_value = mock_job
    
    response = client.get(f"/jobs/{job_id}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"
    assert response.json()["id"] == job_id
