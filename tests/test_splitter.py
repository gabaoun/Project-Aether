import pytest
from typing import List
from llama_index.core.schema import Document
from llama_index.core.embeddings import BaseEmbedding
from src.core.splitter import SemanticDoubleMergingSplitter

@pytest.fixture
def mock_embed_model():
    """Mock embedding model."""
    class MockEmbedding(BaseEmbedding):
        def __init__(self, **kwargs):
            super().__init__(model_name="mock-model", **kwargs)
        def _get_query_embedding(self, query: str): return [0.1] * 384
        def _get_text_embedding(self, text: str): return [0.1] * 384
        def _get_text_embeddings(self, texts: List[str]): return [[0.1] * 384 for _ in texts]
        async def _aget_query_embedding(self, query: str): return [0.1] * 384
        async def _aget_text_embedding(self, text: str): return [0.1] * 384

    return MockEmbedding()

def test_semantic_double_merging_logic(mock_embed_model, mocker):
    """
    Validates if the splitter merges small chunks.
    """
    splitter = SemanticDoubleMergingSplitter(
        embed_model=mock_embed_model,
        min_chunk_size=100
    )
    
    node1 = Document(text="A" * 60)
    node2 = Document(text="B" * 50)
    node3 = Document(text="C" * 110)

    mocker.patch(
        'llama_index.core.node_parser.SemanticSplitterNodeParser.get_nodes_from_documents',
        return_value=[node1, node2, node3]
    )

    doc = Document(text="Irrelevant")
    nodes = list(splitter.get_nodes_generator([doc]))

    assert len(nodes) == 2
    assert "A" * 60 in nodes[0].get_content()
    assert "B" * 50 in nodes[0].get_content()
    assert nodes[1].get_content() == node3.get_content()

def test_generator_type(mock_embed_model):
    splitter = SemanticDoubleMergingSplitter(embed_model=mock_embed_model)
    doc = Document(text="Some text")
    gen = splitter.get_nodes_generator([doc])
    assert hasattr(gen, '__next__')
