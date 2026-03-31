import pytest
from unittest.mock import MagicMock
from src.pipeline.retrieval import RetrievalWorkflow, QueryTransformedEvent
from llama_index.core.workflow import StartEvent, StopEvent

@pytest.mark.asyncio
async def test_retrieval_cache_hit(mocker):
    mock_chroma = MagicMock()
    wf = RetrievalWorkflow(chroma_service=mock_chroma, reranker=None)
    mock_ctx = MagicMock()
    
    # Mock cache hit
    mocker.patch.object(wf.cache, 'get_cache', return_value="Cached Answer")
    
    # Execution
    ev = StartEvent(query="What is Aether?")
    result = await wf.process_start(mock_ctx, ev)
    
    # Validation
    assert isinstance(result, StopEvent)
    assert result.result["answer"] == "Cached Answer"
    assert result.result["from_cache"] is True
    mock_ctx.send_event.assert_called()

@pytest.mark.asyncio
async def test_retrieval_query_transformation(mocker):
    mock_chroma = MagicMock()
    wf = RetrievalWorkflow(chroma_service=mock_chroma, reranker=None)
    mock_ctx = MagicMock()
    
    # Mock cache miss
    mocker.patch.object(wf.cache, 'get_cache', return_value=None)
    
    # Mock LLM for HyDE
    mock_llm_response = MagicMock()
    mock_llm_response.text = "Hypothetical document content"
    mocker.patch.object(wf, '_call_llm_with_retry', return_value=mock_llm_response)
    
    # Execution
    ev = StartEvent(query="New Query")
    result = await wf.process_start(mock_ctx, ev)
    
    # Validation
    assert isinstance(result, QueryTransformedEvent)
    assert result.query_bundle.query_str == "New Query"
    assert "Hypothetical document content" in result.query_bundle.custom_embedding_strs
    mock_ctx.send_event.assert_called()
