import pytest
from unittest.mock import MagicMock, AsyncMock
from llama_index.core.schema import Document
from src.pipeline.ingestion import IngestionWorkflow, DocumentsLoadedEvent
from llama_index.core.workflow import StartEvent

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
