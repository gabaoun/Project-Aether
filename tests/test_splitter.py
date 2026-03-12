import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import Document
from src.processing.splitter import SemanticDoubleMergingSplitter

@pytest.fixture
def mock_embed_model():
    """Creates a mock for the embedding model to avoid costs and latency."""
    embed_model = MagicMock()
    # Mocking necessary methods for SemanticSplitterNodeParser
    embed_model.get_text_embedding.side_effect = lambda x: [0.1] * 384
    embed_model.get_text_embedding_batch.side_effect = lambda x: [[0.1] * 384 for _ in x]
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
    
    # Mocking the base class behavior (Pass 1)
    # We simulate that the base splitter returned 3 nodes: two small ones and one normal
    node1 = MagicMock(spec=Document)
    node1.get_content.return_value = "Short text." # 11 chars
    
    node2 = MagicMock(spec=Document)
    node2.get_content.return_value = "Another short bit." # 18 chars
    
    node3 = MagicMock(spec=Document)
    node3.get_content.return_value = "This is a much longer text that should exceed the minimum chunk size threshold for merging logic." # ~100 chars

    # Inject mock into super().get_nodes_from_documents via mocker
    mocker.patch(
        'llama_index.core.node_parser.SemanticSplitterNodeParser.get_nodes_from_documents',
        return_value=[node1, node2, node3]
    )

    # Execution
    doc = Document(text="Irrelevant for this mock test")
    nodes = list(splitter.get_nodes_generator([doc]))

    # Validation
    # node1 and node2 should have been merged because len(node1) < 100
    # Final result should have 2 nodes (Merge of 1+2 and the isolated 3)
    assert len(nodes) == 2
    assert "Short text." in nodes[0].get_content()
    assert "Another short bit." in nodes[0].get_content()
    assert nodes[1].get_content() == node3.get_content()

def test_generator_memory_efficiency(mock_embed_model):
    """Ensures the splitter returns a generator and not a full list immediately."""
    splitter = SemanticDoubleMergingSplitter(embed_model=mock_embed_model)
    doc = Document(text="Some text")
    
    gen = splitter.get_nodes_generator([doc])
    
    # Verify it's a Generator type
    assert hasattr(gen, '__next__')
