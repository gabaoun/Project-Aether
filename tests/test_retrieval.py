from unittest.mock import AsyncMock, MagicMock

import pytest
import redis
from llama_index.core.workflow import StartEvent, StopEvent

from src.pipeline.retrieval import QueryTransformedEvent, RetrievalWorkflow


@pytest.fixture(autouse=True)
def mock_redis(mocker):
    mock_r = MagicMock()
    mock_r.ping.side_effect = redis.ConnectionError("No Redis")
    mocker.patch("redis.Redis", return_value=mock_r)


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
    mock_ctx.write_event_to_stream.assert_called()

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
    mock_ctx.write_event_to_stream.assert_called()


@pytest.mark.asyncio
async def test_full_workflow_run_is_a_valid_event_graph(mocker):
    # Exercises the real wf.run() entrypoint (not individual step methods
    # directly, like the tests above). llama-index-workflows validates the
    # step event graph on .run() - a step declaring it can produce an event
    # type nothing consumes raises WorkflowValidationError before any step
    # even executes. This is exactly the class of bug that shipped
    # (StreamingStatusEvent declared in return-type unions but never
    # actually returned) without the tests above ever catching it.
    mock_chroma = MagicMock()
    mock_chroma.hybrid_search = AsyncMock(return_value=[
        {"id": "doc1_chunk_0", "content": "Gabriel has C++ experience.", "metadata": {}, "score": 0.1},
    ])
    wf = RetrievalWorkflow(chroma_service=mock_chroma, reranker=None)

    mock_llm_response = MagicMock()
    mock_llm_response.text = "YES - Gabriel has C++ experience."
    mocker.patch.object(wf, '_call_llm_with_retry', return_value=mock_llm_response)
    mocker.patch.object(wf.cache, 'get_cache', return_value=None)
    mocker.patch.object(wf.cache, 'set_cache', return_value=None)

    result = await wf.run(query="What C++ experience does Gabriel have?")

    assert result["from_cache"] is False
    assert "C++" in result["answer"]
