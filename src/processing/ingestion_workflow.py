import os
from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from src.processing.splitter import SemanticDoubleMergingSplitter
from src.utils.config import settings
from src.utils.token_counter import TokenCounter
from src.utils.pii_masker import PIIMasker
from src.retrieval.semantic_cache import SemanticCache
from src.utils.logger import logger
from src.core.exceptions import IngestionException

# Custom events for the workflow
class DocumentsLoadedEvent(Event):
    documents: List[Document]

class NodesCreatedEvent(Event):
    nodes: List[Document]

class MetadataEnrichedEvent(Event):
    nodes: List[Document]

class IngestionWorkflow(Workflow):
    """
    An event-driven workflow for ingesting, chunking, enriching, and indexing documents.
    Leverages LlamaIndex Workflows, Tenacity for retries, and async processing for efficiency.
    
    Attributes:
        qdrant_client (QdrantClient): The client for Qdrant vector database.
        collection_name (str): The name of the collection to index documents into.
        embed_model (HuggingFaceEmbedding): The embedding model used for vectorization.
        node_parser (SemanticDoubleMergingSplitter): Custom semantic node splitter.
        token_counter (TokenCounter): Utility to track LLM token usage and costs.
        pii_masker (PIIMasker): Utility to mask PII from ingested documents.
        cache (SemanticCache): Redis-based semantic cache for retrieval optimization.
    """
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, **kwargs: dict) -> None:
        """Initializes the IngestionWorkflow with necessary components and clients."""
        super().__init__(**kwargs)
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.node_parser = SemanticDoubleMergingSplitter(
            embed_model=self.embed_model,
            min_chunk_size=500
        )
        self.token_counter = TokenCounter()
        self.pii_masker = PIIMasker()
        self.cache = SemanticCache()

    @step
    async def load_documents(self, ev: StartEvent) -> DocumentsLoadedEvent:
        """
        Loads documents from the specified input directory and asynchronously masks PII.
        
        Args:
            ev (StartEvent): The start event containing 'input_dir'.
            
        Returns:
            DocumentsLoadedEvent: An event containing the loaded and masked documents.
            
        Raises:
            IngestionException: If 'input_dir' is missing from the event.
        """
        from llama_index.core import SimpleDirectoryReader
        input_dir = ev.get("input_dir")
        if not input_dir:
            raise IngestionException("input_dir must be provided in StartEvent", status_code=400)
            
        reader = SimpleDirectoryReader(input_dir=input_dir)
        documents = reader.load_data()
        
        # Async PII Masking
        original_texts = [doc.text for doc in documents]
        masked_texts = await self.pii_masker.mask_documents_async(original_texts)
        
        for doc, masked_text in zip(documents, masked_texts):
            doc.text = masked_text
            
        logger.info(f"[INGESTION] Loaded and asynchronously masked {len(documents)} documents.")
        return DocumentsLoadedEvent(documents=documents)

    @step
    async def chunk_documents(self, ev: DocumentsLoadedEvent) -> NodesCreatedEvent:
        """
        Chunks documents into semantic nodes using a memory-efficient generator.
        
        Args:
            ev (DocumentsLoadedEvent): The event containing documents to chunk.
            
        Returns:
            NodesCreatedEvent: An event containing the resulting semantic nodes.
        """
        # Utilizing generator for memory efficiency
        nodes = list(self.node_parser.get_nodes_generator(ev.documents))
        logger.info(f"[INGESTION] Created {len(nodes)} semantic nodes using Generator.")
        return NodesCreatedEvent(nodes=nodes)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm_with_retry(self, llm: "OpenAI", prompt: str) -> "Response": # type: ignore
        """
        A resilient wrapper for async LLM calls with exponential backoff retries.
        
        Args:
            llm: The initialized OpenAI LLM instance.
            prompt (str): The prompt to send to the LLM.
            
        Returns:
            Response: The response from the LLM.
        """
        response = await llm.acomplete(prompt)
        self.token_counter.log_cost("MetadataEnrichment", prompt, response.text)
        return response

    @step
    async def enrich_metadata(self, ev: NodesCreatedEvent) -> MetadataEnrichedEvent:
        """
        Enriches nodes by generating summaries and hypothetical questions via LLM.
        
        Args:
            ev (NodesCreatedEvent): The event containing raw nodes.
            
        Returns:
            MetadataEnrichedEvent: An event containing nodes with enriched metadata.
        """
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
        
        enriched_nodes = []
        for node in ev.nodes:
            prompt = (
                f"For the following chunk of text, generate a brief summary and 3 hypothetical questions it answers.\n"
                f"Format: Summary: [summary]\nQuestions: 1. [q1] 2. [q2] 3. [q3]\n\n"
                f"Text:\n{node.get_content()}"
            )
            response = await self._call_llm_with_retry(llm, prompt)
            node.metadata["enrichment"] = response.text
            enriched_nodes.append(node)
            
        logger.info("[INGESTION] Metadata enrichment complete.")
        return MetadataEnrichedEvent(nodes=enriched_nodes)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _persist_with_retry(self, nodes: List[Document]) -> VectorStoreIndex:
        """
        A resilient wrapper for persisting nodes to Qdrant with exponential backoff.
        
        Args:
            nodes (List[Document]): The nodes to persist.
            
        Returns:
            VectorStoreIndex: The created or updated vector store index.
        """
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            enable_hybrid=True
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            embed_model=self.embed_model
        )

    @step
    async def persist_to_qdrant(self, ev: MetadataEnrichedEvent) -> StopEvent:
        """
        Persists the enriched nodes to the Qdrant vector database and invalidates the cache.
        
        Args:
            ev (MetadataEnrichedEvent): The event containing enriched nodes.
            
        Returns:
            StopEvent: The final event containing the vector store index.
            
        Raises:
            IngestionException: If persistence fails after all retries.
        """
        try:
            index = self._persist_with_retry(ev.nodes)
            logger.info(f"[INGESTION] Successfully indexed {len(ev.nodes)} nodes in Qdrant '{self.collection_name}'.")
            
            # Smart Cache Invalidation
            logger.info("[INGESTION] Invalidating Semantic Cache due to new data ingestion.")
            self.cache.invalidate_cache()
            
            return StopEvent(result=index)
        except Exception as e:
            logger.error(f"[INGESTION] Failed to index nodes in Qdrant after retries: {e}", exc_info=True)
            raise IngestionException(f"Failed to persist nodes: {e}")
