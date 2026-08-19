
import re
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.core.pii import PIIMasker
from src.models.exceptions import IngestionException
from src.services.chroma import ChromaService
from src.services.groq_client import LightweightGroqLLM
from src.services.neo4j import Neo4jService
from src.services.redis import SemanticCache
from src.utils.logger import logger
from src.utils.token_counter import TokenCounter

# Source docs can wrap internal/strategy notes (not meant for the public
# recruiter-facing bot) in <!-- PRIVATE:START --> ... <!-- PRIVATE:END -->
# comments; those spans are dropped before chunking/embedding so they never
# reach the retrievable corpus, while staying intact for other tooling that
# reads the raw file directly.
_PRIVATE_BLOCK_RE = re.compile(
    r'<!--\s*PRIVATE:START\s*-->.*?<!--\s*PRIVATE:END\s*-->',
    re.IGNORECASE | re.DOTALL,
)


class DocumentsLoadedEvent(Event):
    documents: list[Document]

class NodesCreatedEvent(Event):
    nodes: list[BaseNode]

class MetadataEnrichedEvent(Event):
    nodes: list[BaseNode]

class IngestionWorkflow(Workflow):
    """
    Workflow for ingesting, chunking, and indexing documents using Chroma Cloud.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chroma_service = ChromaService()
        self.neo4j_service = Neo4jService()
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
        
        # Strip PRIVATE-marked spans, then PII-mask what's left
        stripped_texts = [_PRIVATE_BLOCK_RE.sub("", doc.text) for doc in documents]
        masked_texts = await self.pii_masker.mask_documents_async(stripped_texts)
        
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
        nodes = await self.node_parser.aget_nodes_from_documents(ev.documents)
        logger.info(f"[INGESTION] Created {len(nodes)} semantic nodes.")
        return NodesCreatedEvent(nodes=nodes)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm_with_retry(self, llm, prompt: str):
        response = await llm.acomplete(prompt)
        self.token_counter.log_cost("MetadataEnrichment", prompt, response.text)
        return response

    @step
    async def enrich_metadata(self, ev: NodesCreatedEvent) -> MetadataEnrichedEvent:
        # LightweightGroqLLM, not llama_index.llms.groq.Groq: only .acomplete()
        # is used here, and the llama_index wrapper drags in transformers/torch
        # for no benefit in this narrow usage - see src/services/groq_client.py.
        llm = LightweightGroqLLM(model="openai/gpt-oss-20b", api_key=settings.groq_api_key)
        
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
            
            await self.chroma_service.clear_collection()
            await self.chroma_service.upsert_documents(docs_to_upsert)
            logger.info(f"[INGESTION] Indexed {len(ev.nodes)} nodes in Chroma Cloud.")

            if self.neo4j_service.is_enabled():
                try:
                    from llama_index.llms.groq import Groq
                    llm = Groq(model="openai/gpt-oss-20b", api_key=settings.groq_api_key)
                    await self.neo4j_service.extract_and_index_nodes(ev.nodes, llm=llm)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"[INGESTION] Neo4j graph indexing failed: {e}")

            self.cache.invalidate_cache()
            return StopEvent(result=self.chroma_service)
        except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
            logger.error(f"[INGESTION] Failed to index nodes in Chroma: {e}")
            raise IngestionException(f"Failed to persist nodes: {e}")

