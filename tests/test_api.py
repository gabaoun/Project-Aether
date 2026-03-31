import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
from src.api.app import app
import src.api.app as api_app

client = TestClient(app)

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
    # Before initialization, it should return 503
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
