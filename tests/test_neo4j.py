from unittest.mock import AsyncMock, MagicMock

import pytest
import redis
from llama_index.core.schema import NodeWithScore, TextNode

from src.pipeline.retrieval import QueryTransformedEvent, RetrievalWorkflow
from src.services.neo4j import Neo4jService


@pytest.fixture(autouse=True)
def mock_redis(mocker):
    mock_r = MagicMock()
    mock_r.ping.side_effect = redis.ConnectionError("No Redis")
    mocker.patch("redis.Redis", return_value=mock_r)


@pytest.mark.asyncio
async def test_neo4j_service_disabled_by_default(mocker):
    mocker.patch("src.services.neo4j.settings.enable_neo4j", False)
    service = Neo4jService()

    assert service.is_enabled() is False
    assert service.get_graph_store() is None

    result = await service.retrieve("test query")
    assert result == []

    # extract_and_index_nodes should do nothing and return without error
    nodes = [TextNode(text="Test node")]
    await service.extract_and_index_nodes(nodes)


@pytest.mark.asyncio
async def test_neo4j_service_enabled(mocker):
    mocker.patch("src.services.neo4j.settings.enable_neo4j", True)
    mock_store = MagicMock()
    mocker.patch(
        "llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore",
        return_value=mock_store,
    )

    service = Neo4jService(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
    )

    assert service.is_enabled() is True
    assert service.get_graph_store() == mock_store


@pytest.mark.asyncio
async def test_neo4j_extract_and_index_nodes(mocker):
    mocker.patch("src.services.neo4j.settings.enable_neo4j", True)
    mocker.patch("src.services.neo4j.SimpleLLMPathExtractor")
    mock_store = MagicMock()
    mocker.patch(
        "llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore",
        return_value=mock_store,
    )

    mock_index = MagicMock()
    mock_index.ainsert_nodes = AsyncMock()
    mocker.patch(
        "src.services.neo4j.PropertyGraphIndex.from_existing",
        return_value=mock_index,
    )

    service = Neo4jService()
    nodes = [TextNode(text="Gabriel is a software engineer.")]
    mock_llm = MagicMock()

    await service.extract_and_index_nodes(nodes, llm=mock_llm)

    mock_index.ainsert_nodes.assert_called_once()


@pytest.mark.asyncio
async def test_neo4j_retrieve(mocker):
    mocker.patch("src.services.neo4j.settings.enable_neo4j", True)
    mocker.patch("src.services.neo4j.LLMSynonymRetriever")
    mock_store = MagicMock()
    mocker.patch(
        "llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore",
        return_value=mock_store,
    )

    expected_node = NodeWithScore(
        node=TextNode(text="Knowledge Graph Entity Node"), score=0.9
    )
    mock_retriever = MagicMock()
    mock_retriever.aretrieve = AsyncMock(return_value=[expected_node])

    mock_index = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever
    mocker.patch(
        "src.services.neo4j.PropertyGraphIndex.from_existing",
        return_value=mock_index,
    )

    service = Neo4jService()
    mock_llm = MagicMock()

    results = await service.retrieve("Who is Gabriel?", llm=mock_llm)

    assert len(results) == 1
    assert results[0].node.get_content() == "Knowledge Graph Entity Node"


@pytest.mark.asyncio
async def test_neo4j_connection_failure_degrades_gracefully(mocker):
    mocker.patch("src.services.neo4j.settings.enable_neo4j", True)
    mocker.patch(
        "llama_index.graph_stores.neo4j.Neo4jPropertyGraphStore",
        side_effect=Exception("Connection refused"),
    )

    service = Neo4jService()

    assert service.is_enabled() is False
    results = await service.retrieve("query")
    assert results == []


@pytest.mark.asyncio
async def test_retrieval_workflow_hybrid_search_with_neo4j(mocker):
    mock_chroma = MagicMock()
    mock_chroma.hybrid_search = AsyncMock(
        return_value=[
            {
                "id": "vector_doc_1",
                "content": "Dense vector content from Chroma.",
                "metadata": {},
                "score": 0.85,
            }
        ]
    )

    mock_neo4j = MagicMock()
    mock_neo4j.is_enabled.return_value = True
    graph_node = NodeWithScore(
        node=TextNode(text="Graph node from Neo4j.", id_="graph_doc_1"), score=0.9
    )
    mock_neo4j.retrieve = AsyncMock(return_value=[graph_node])

    wf = RetrievalWorkflow(chroma_service=mock_chroma, neo4j_service=mock_neo4j)
    mock_ctx = MagicMock()

    from llama_index.core import QueryBundle

    ev = QueryTransformedEvent(
        query_bundle=QueryBundle(query_str="Test Query"), loops=0
    )
    result = await wf.retrieve_context(mock_ctx, ev)

    assert len(result.nodes) == 2
    contents = [n.get_content() for n in result.nodes]
    assert "Dense vector content from Chroma." in contents
    assert "Graph node from Neo4j." in contents
