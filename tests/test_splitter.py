import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import Document, BaseNode
from src.processing.splitter import SemanticDoubleMergingSplitter

class MockEmbedding:
    def get_text_embedding(self, text):
        return [0.1] * 128
    def get_text_embeddings(self, texts):
        return [[0.1] * 128 for _ in texts]
    def get_query_embedding(self, query):
        return [0.1] * 128
    async def aget_text_embedding(self, text):
        return [0.1] * 128
    async def aget_query_embedding(self, query):
        return [0.1] * 128
    @property
    def model_name(self):
        return "mock_model"
    @property
    def embed_batch_size(self):
        return 10
    def __call__(self, *args, **kwargs):
        pass

@pytest.fixture
def mock_embed_model():
    return MockEmbedding()

def test_double_merging_generator(mock_embed_model, mocker):
    # Mocking the parent class get_nodes_from_documents to avoid complex embedding logic
    # We want to test pure double merging logic, isolated from external APIs
    
    node1 = BaseNode(text="Short")
    node2 = BaseNode(text="This is a slightly longer sentence that should not be merged ideally if threshold was lower.")
    node3 = BaseNode(text="Short2")
    
    mocker.patch(
        'llama_index.core.node_parser.SemanticSplitterNodeParser.get_nodes_from_documents',
        return_value=[node1, node2, node3]
    )

    splitter = SemanticDoubleMergingSplitter(
        embed_model=mock_embed_model,
        min_chunk_size=10, 
        buffer_size=1,
        breakpoint_percentile_threshold=95
    )
    
    doc = Document(text="Mock text")
    
    generator = splitter.get_nodes_generator([doc])
    nodes = list(generator)
    
    # "Short" (len < 10) should merge with the next long node
    assert len(nodes) == 2
    assert "Short\nThis is a slightly longer" in nodes[0].get_content()
    assert "Short2" in nodes[1].get_content()
