from typing import List
from llama_index.core import Document
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.core.node_parser import SentenceSplitter
from src.config.settings import settings
from src.utils.token_counter import TokenCounter
from src.core.pii import PIIMasker
from src.services.redis import SemanticCache
from src.services.chroma import ChromaService
from src.utils.logger import logger
from src.models.exceptions import IngestionException
class DocumentsLoadedEvent(Event):
    documents: List[Document]

class NodesCreatedEvent(Event):
    nodes: List[Document]

class MetadataEnrichedEvent(Event):
    nodes: List[Document]

class IngestionWorkflow(Workflow):
    """
    Workflow for ingesting, chunking, and indexing documents using Chroma Cloud.
    """
    def __init__(self, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.chroma_service = ChromaService()
        self.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20
        )
        self.token_counter = TokenCounter()
        self.pii_masker = PIIMasker()
        self.cache = SemanticCache()

    @step
    async def load_documents(self, ev: StartEvent) -> DocumentsLoadedEvent:
        from llama_index.core import SimpleDirectoryReader
        input_dir = ev.get("input_dir")
        if not input_dir:
            raise IngestionException("input_dir must be provided in StartEvent", status_code=400)
            
        reader = SimpleDirectoryReader(input_dir=input_dir)
        documents = reader.load_data()
        
        # Async PII Masking
        original_texts = [doc.text for doc in documents]
        masked_texts = await self.pii_masker.mask_documents_async(original_texts)
        
        masked_documents = []
        for doc, masked_text in zip(documents, masked_texts):
            new_doc = Document(
                text=masked_text,
                metadata=doc.metadata,
                id_=doc.id_
            )
            masked_documents.append(new_doc)
            
        logger.info(f"[INGESTION] Loaded and masked {len(masked_documents)} documents.")
        return DocumentsLoadedEvent(documents=masked_documents)

    @step
    async def chunk_documents(self, ev: DocumentsLoadedEvent) -> NodesCreatedEvent:
        nodes = list(self.node_parser.get_nodes_generator(ev.documents))
        logger.info(f"[INGESTION] Created {len(nodes)} semantic nodes.")
        return NodesCreatedEvent(nodes=nodes)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm_with_retry(self, llm, prompt: str):
        response = await llm.acomplete(prompt)
        self.token_counter.log_cost("MetadataEnrichment", prompt, response.text)
        return response

    @step
    async def enrich_metadata(self, ev: NodesCreatedEvent) -> MetadataEnrichedEvent:
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
        
        enriched_nodes = []
        for node in ev.nodes:
            prompt = (
                f"For the following text, generate a brief summary and 3 hypothetical questions it answers.\n"
                f"Format: Summary: [summary]\nQuestions: 1. [q1] 2. [q2] 3. [q3]\n\n"
                f"Text:\n{node.get_content()}"
            )
            response = await self._call_llm_with_retry(llm, prompt)
            node.metadata["enrichment"] = response.text
            enriched_nodes.append(node)
            
        logger.info("[INGESTION] Metadata enrichment complete.")
        return MetadataEnrichedEvent(nodes=enriched_nodes)

    @step
    async def persist_to_chroma(self, ev: MetadataEnrichedEvent) -> StopEvent:
        try:
            docs_to_upsert = []
            for node in ev.nodes:
                docs_to_upsert.append({
                    "id": node.node_id,
                    "text": node.get_content(),
                    "metadata": node.metadata
                })
            
            await self.chroma_service.upsert_documents(docs_to_upsert)
            logger.info(f"[INGESTION] Indexed {len(ev.nodes)} nodes in Chroma Cloud.")
            self.cache.invalidate_cache()
            return StopEvent(result=self.chroma_service)
        except Exception as e:
            logger.error(f"[INGESTION] Failed to index nodes in Chroma: {e}")
            raise IngestionException(f"Failed to persist nodes: {e}")

