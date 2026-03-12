import pytest
from typing import List
from unittest.mock import MagicMock
from llama_index.core.schema import Document
from llama_index.core.embeddings import BaseEmbedding
from src.processing.splitter import SemanticDoubleMergingSplitter

@pytest.fixture
def mock_embed_model():
    """Creates a mock for the embedding model to avoid costs and latency."""
    class MockEmbedding(BaseEmbedding):
        def __init__(self, **kwargs):
            super().__init__(model_name="mock-model", **kwargs)
        def _get_query_embedding(self, query: str): return [0.1] * 384
        def _get_text_embedding(self, text: str): return [0.1] * 384
        def _get_text_embeddings(self, texts: List[str]): return [[0.1] * 384 for _ in texts]
        async def _aget_query_embedding(self, query: str): return [0.1] * 384
        async def _aget_text_embedding(self, text: str): return [0.1] * 384

    embed_model = MockEmbedding()
    return embed_model

def test_semantic_double_merging_logic(mock_embed_model, mocker):
    """
    Validates if the splitter merges small chunks (below min_chunk_size)
    and maintains content integrity.
    """
    # Configuration: min_chunk_size of 100 characters
    splitter = SemanticDoubleMergingSplitter(
        embed_model=mock_embed_model,
        min_chunk_size=100
    )
    
    # Use real Document objects to ensure content updates (set_content) work as expected
    node1 = Document(text="A" * 60) # 60 chars
    node2 = Document(text="B" * 50) # 50 chars. node1 + node2 = 110 (> 100)
    node3 = Document(text="C" * 110) # 110 chars. Already big enough.

    # Inject mock into the base class method. 
    # We patch it in the splitter module where it's imported.
    mocker.patch(
        'src.processing.splitter.SemanticSplitterNodeParser.get_nodes_from_documents',
        return_value=[node1, node2, node3]
    )

    # Execution
    doc = Document(text="Irrelevant for this mock test")
    nodes = list(splitter.get_nodes_generator([doc]))

    # Validation
    # node1 and node2 should have been merged because len(node1) < 100.
    # After merging node2, node1's length becomes 111 (60 + 1 (newline) + 50).
    # Since 111 > 100, the next iteration (node3) will yield the merged node1 and start fresh.
    # Final result should have 2 nodes: (node1+node2) and (node3)
    assert len(nodes) == 2
    assert "A" * 60 in nodes[0].get_content()
    assert "B" * 50 in nodes[0].get_content()
    assert nodes[1].get_content() == node3.get_content()

def test_generator_memory_efficiency(mock_embed_model):
    """Ensures the splitter returns a generator and not a full list immediately."""
    splitter = SemanticDoubleMergingSplitter(embed_model=mock_embed_model)
    doc = Document(text="Some text")
    
    gen = splitter.get_nodes_generator([doc])
    
    # Verify it's a Generator type
    assert hasattr(gen, '__next__')
