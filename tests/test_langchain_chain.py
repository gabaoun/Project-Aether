from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.exceptions import RetrievalException
from src.pipeline.langchain_chain import LangChainRAGChain


@pytest.fixture
def mock_chroma():
    mock = MagicMock()
    mock.hybrid_search = AsyncMock(return_value=[
        {"id": "doc1", "content": "Aether is a RAG engine.", "metadata": {"file_name": "readme.md"}, "score": 0.1}
    ])
    return mock


@pytest.mark.asyncio
async def test_aquery_returns_answer_and_sources(mocker, mock_chroma):
    chain = LangChainRAGChain(chroma_service=mock_chroma)
    mocker.patch.object(chain, "_invoke_with_retry", AsyncMock(return_value="Aether is a RAG engine."))

    result = await chain.aquery("What is Aether?")

    assert result["answer"] == "Aether is a RAG engine."
    assert result["from_cache"] is False
    assert result["source_nodes"] == [{"file_name": "readme.md"}]
    mock_chroma.hybrid_search.assert_awaited()


@pytest.mark.asyncio
async def test_aquery_rejects_empty_query(mock_chroma):
    chain = LangChainRAGChain(chroma_service=mock_chroma)

    with pytest.raises(RetrievalException):
        await chain.aquery("")


@pytest.mark.asyncio
async def test_aquery_wraps_retrieval_failure(mocker, mock_chroma):
    mock_chroma.hybrid_search = AsyncMock(side_effect=RuntimeError("Chroma unreachable"))
    chain = LangChainRAGChain(chroma_service=mock_chroma)

    with pytest.raises(RetrievalException):
        await chain.aquery("What is Aether?")


@pytest.mark.asyncio
async def test_retrieve_joins_context_and_falls_back_when_empty(mock_chroma):
    chain = LangChainRAGChain(chroma_service=mock_chroma)

    context = await chain._retrieve("What is Aether?")
    assert context == "Aether is a RAG engine."

    mock_chroma.hybrid_search = AsyncMock(return_value=[])
    empty_context = await chain._retrieve("Unanswerable question")
    assert empty_context == "No relevant context found."
