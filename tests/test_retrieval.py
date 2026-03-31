import pytest
from unittest.mock import MagicMock, AsyncMock
from src.pipeline.retrieval import RetrievalWorkflow, QueryTransformedEvent
from llama_index.core.workflow import StartEvent, StopEvent

@pytest.mark.asyncio
async def test_retrieval_cache_hit(mocker):
    mock_chroma = MagicMock()
    wf = RetrievalWorkflow(chroma_service=mock_chroma)
    
    # Mock cache hit
    mocker.patch.object(wf.cache, 'get_cache', return_value="Cached Answer")
    
    # Execution
    ev = StartEvent(query="What is Aether?")
    result = await wf.process_start(ev)
    
    # Validation
    assert isinstance(result, StopEvent)
    assert result.result["answer"] == "Cached Answer"
    assert result.result["from_cache"] is True

@pytest.mark.asyncio
async def test_retrieval_query_transformation(mocker):
    mock_chroma = MagicMock()
    wf = RetrievalWorkflow(chroma_service=mock_chroma)
    
    # Mock cache miss
    mocker.patch.object(wf.cache, 'get_cache', return_value=None)
    
    # Mock LLM for HyDE
    mock_llm_response = MagicMock()
    mock_llm_response.text = "Hypothetical document content"
    mocker.patch.object(wf, '_call_llm_with_retry', return_value=mock_llm_response)
    
    # Execution
    ev = StartEvent(query="New Query")
    result = await wf.process_start(ev)
    
    # Validation
    assert isinstance(result, QueryTransformedEvent)
    assert result.query_bundle.query_str == "New Query"
    assert "Hypothetical document content" in result.query_bundle.custom_embedding_strs
