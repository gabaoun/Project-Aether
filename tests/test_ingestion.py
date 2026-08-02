from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import BaseNode, Document
from llama_index.core.workflow import StartEvent

from src.pipeline.ingestion import (
    DocumentsLoadedEvent,
    IngestionWorkflow,
    NodesCreatedEvent,
)


@pytest.mark.asyncio
async def test_load_documents_and_mask_pii(mocker):
    # Mocking ChromaService and other dependencies
    mocker.patch('src.pipeline.ingestion.ChromaService')
    wf = IngestionWorkflow()
    
    # Mock SimpleDirectoryReader
    mock_reader = MagicMock()
    mock_reader.load_data.return_value = [
        Document(text="Contact me at test@example.com or 123-456-7890")
    ]
    mocker.patch('llama_index.core.SimpleDirectoryReader', return_value=mock_reader)
    
    # Execution
    ev = StartEvent(input_dir="./data")
    result = await wf.load_documents(ev)
    
    # Validation
    assert isinstance(result, DocumentsLoadedEvent)
    assert len(result.documents) == 1
    assert "[EMAIL]" in result.documents[0].text
    assert "[PHONE]" in result.documents[0].text
    assert "test@example.com" not in result.documents[0].text


@pytest.mark.asyncio
async def test_chunk_documents_uses_the_real_node_parser_api(mocker):
    # No mocking of the node parser here - this exercises the actual
    # SentenceSplitter API surface, which is exactly what let a call to a
    # nonexistent method (get_nodes_generator) ship undetected before.
    mocker.patch('src.pipeline.ingestion.ChromaService')
    wf = IngestionWorkflow()

    long_text = "This is a sentence. " * 200  # long enough to actually split into multiple nodes
    ev = DocumentsLoadedEvent(documents=[Document(text=long_text)])

    result = await wf.chunk_documents(ev)

    assert isinstance(result, NodesCreatedEvent)
    assert len(result.nodes) > 0
    assert all(isinstance(node, BaseNode) for node in result.nodes)
