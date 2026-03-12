from typing import List, Optional, AsyncGenerator, Union
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.postprocessor import LongContextReorder
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import settings
from src.retrieval.semantic_cache import SemanticCache
from src.utils.token_counter import TokenCounter
from src.utils.logger import logger
from src.core.exceptions import RetrievalException

# Custom Events
class StreamingStatusEvent(Event):
    status: str

class QueryTransformedEvent(Event):
    query_bundle: QueryBundle
    loops: int
    tenant_id: Optional[str]

class ContextRetrievedEvent(Event):
    nodes: List[NodeWithScore]
    query_bundle: QueryBundle
    loops: int
    tenant_id: Optional[str]

class RelevanceJudgedEvent(Event):
    is_relevant: bool
    nodes: List[NodeWithScore]
    query_bundle: QueryBundle
    tenant_id: Optional[str]

class RetrievalWorkflow(Workflow):
    """
    An event-driven workflow for complex RAG retrieval.
    Implements query decomposition, HyDE, relevance judgment (CoT), reranking, and cache checks.
    
    Attributes:
        index (VectorStoreIndex): The vector store index to search against.
        llm (OpenAI): The LLM used for generation and evaluation.
        reranker (FlagEmbeddingReranker): The cross-encoder reranker for initial results.
        reorder (LongContextReorder): Utility to reorder nodes for the 'Lost in the Middle' problem.
        cache (SemanticCache): The semantic cache layer.
        token_counter (TokenCounter): Utility to track token usage.
    """
    def __init__(self, index: VectorStoreIndex, **kwargs: dict) -> None:
        """Initializes the RetrievalWorkflow with the index and processing components."""
        super().__init__(**kwargs)
        self.index = index
        self.llm = OpenAI(model="gpt-4o", api_key=settings.openai_api_key)
        self.reranker = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-v2-m3",
            top_n=5
        )
        self.reorder = LongContextReorder()
        self.cache = SemanticCache()
        self.token_counter = TokenCounter(model_name="gpt-4o")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm_with_retry(self, prompt: str) -> "Response": # type: ignore
        """
        A resilient wrapper for async LLM calls with exponential backoff retries.
        
        Args:
            prompt (str): The prompt to send to the LLM.
            
        Returns:
            Response: The response from the LLM.
        """
        response = await self.llm.acomplete(prompt)
        self.token_counter.log_cost("RetrievalLLM", prompt, response.text)
        return response

    @step
    async def process_start(self, ev: StartEvent) -> Union[QueryTransformedEvent, StopEvent, StreamingStatusEvent]:
        """
        Initiates the retrieval process. Checks cache and performs query decomposition and HyDE.
        
        Args:
            ev (StartEvent): The start event containing 'query' and optional 'tenant_id'.
            
        Returns:
            Union[QueryTransformedEvent, StopEvent, StreamingStatusEvent]: The next event in the workflow.
            
        Raises:
            RetrievalException: If 'query' is missing.
        """
        query_str = ev.get("query")
        tenant_id = ev.get("tenant_id") # Governance / Multi-Tenancy

        if not query_str:
            raise RetrievalException("query must be provided in StartEvent", status_code=400)

        # 1. Semantic Cache Check (Only skip if no tenant is specified, or tenant supports shared cache)
        # Note: In a fully isolated multi-tenant system, Cache keys should include tenant_id. 
        # For simplicity, we cache by query_str, assuming same tenant or shared knowledge.
        cached_answer = self.cache.get_cache(query_str)
        if cached_answer:
            self.send_event(StreamingStatusEvent(status="Cache Hit! Returning fast response..."))
            return StopEvent(result={"answer": cached_answer, "source_nodes": [], "from_cache": True})

        self.send_event(StreamingStatusEvent(status="Decomposing query..."))

        # Decompose
        decomposition_prompt = (
            f"Decompose the following complex query into 2-3 simpler sub-queries if necessary. "
            f"If it's already simple, just return the original query.\n"
            f"Query: {query_str}\n"
            f"Format: List the sub-queries separated by newlines."
        )
        try:
            response = await self._call_llm_with_retry(decomposition_prompt)
            sub_queries = response.text.strip().split("\n")
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error decomposing query: {e}", exc_info=True)
            sub_queries = [query_str]

        self.send_event(StreamingStatusEvent(status="Generating HyDE document..."))
        # HyDE
        hyde_prompt = f"Write a hypothetical document that would answer the following question: {query_str}"
        try:
            hyde_doc = await self._call_llm_with_retry(hyde_prompt)
            custom_embeddings = sub_queries + [hyde_doc.text]
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error generating HyDE document: {e}", exc_info=True)
            custom_embeddings = sub_queries
            
        query_bundle = QueryBundle(query_str=query_str, custom_embedding_strs=custom_embeddings)
        
        return QueryTransformedEvent(query_bundle=query_bundle, loops=0, tenant_id=tenant_id)

    @step
    async def retrieve_context(self, ev: QueryTransformedEvent) -> Union[ContextRetrievedEvent, StreamingStatusEvent]:
        """
        Retrieves context from the vector database using the transformed query bundle.
        
        Args:
            ev (QueryTransformedEvent): The event containing the query bundle.
            
        Returns:
            Union[ContextRetrievedEvent, StreamingStatusEvent]: The event containing retrieved nodes.
        """
        self.send_event(StreamingStatusEvent(status="Retrieving context from Qdrant..."))
        
        filters = None
        if ev.tenant_id:
            filters = MetadataFilters(filters=[ExactMatchFilter(key="tenant_id", value=ev.tenant_id)])
            logger.info(f"[RETRIEVAL] Applying tenant filter: {ev.tenant_id}")
            
        retriever = self.index.as_retriever(similarity_top_k=20, filters=filters)
        nodes = retriever.retrieve(ev.query_bundle)
        
        if not nodes:
            logger.warning("[RETRIEVAL] No nodes retrieved from Qdrant.")
            
        return ContextRetrievedEvent(nodes=nodes, query_bundle=ev.query_bundle, loops=ev.loops, tenant_id=ev.tenant_id)

    @step
    async def judge_relevance(self, ev: ContextRetrievedEvent) -> Union[RelevanceJudgedEvent, QueryTransformedEvent, StreamingStatusEvent]:
        """
        Judges the relevance of retrieved context using Chain of Thought (CoT).
        Refines the query if context is deemed irrelevant.
        
        Args:
            ev (ContextRetrievedEvent): The event containing the retrieved context.
            
        Returns:
            Union[RelevanceJudgedEvent, QueryTransformedEvent, StreamingStatusEvent]: 
            The relevance judgment event or a loop back to query transformation.
        """
        if ev.loops >= 2 or not ev.nodes:
            self.send_event(StreamingStatusEvent(status="Max refinement loops reached or no nodes. Proceeding with best effort..."))
            return RelevanceJudgedEvent(is_relevant=True, nodes=ev.nodes, query_bundle=ev.query_bundle, tenant_id=ev.tenant_id)

        self.send_event(StreamingStatusEvent(status="Judging context relevance (CoT)..."))
        context_text = "\n".join([n.get_content() for n in ev.nodes[:3]])
        
        judge_prompt = (
            f"Query: {ev.query_bundle.query_str}\n"
            f"Context:\n{context_text}\n"
            f"Analyze step-by-step if the context contains enough information to answer the query.\n"
            f"Finally, output a single line with 'VERDICT: YES' or 'VERDICT: NO'."
        )
        
        try:
            response = await self._call_llm_with_retry(judge_prompt)
            is_relevant = "VERDICT: YES" in response.text.upper()
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error judging relevance: {e}", exc_info=True)
            is_relevant = True # Fallback
        
        if not is_relevant:
            self.send_event(StreamingStatusEvent(status="Context irrelevant. Refining query..."))
            refine_prompt = (
                f"The previous search for '{ev.query_bundle.query_str}' returned irrelevant results.\n"
                f"Judge reasoning: {response.text}\n"
                f"Rewrite the query to be more specific."
            )
            try:
                new_query_resp = await self._call_llm_with_retry(refine_prompt)
                new_bundle = QueryBundle(query_str=new_query_resp.text)
                return QueryTransformedEvent(query_bundle=new_bundle, loops=ev.loops + 1, tenant_id=ev.tenant_id)
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error refining query: {e}", exc_info=True)
            
        return RelevanceJudgedEvent(is_relevant=True, nodes=ev.nodes, query_bundle=ev.query_bundle, tenant_id=ev.tenant_id)

    @step
    async def post_process(self, ev: RelevanceJudgedEvent) -> Union[StopEvent, StreamingStatusEvent]:
        """
        Post-processes the context by reranking, reordering, and generating the final answer.
        Updates the semantic cache with the new result.
        
        Args:
            ev (RelevanceJudgedEvent): The event containing the judged context.
            
        Returns:
            Union[StopEvent, StreamingStatusEvent]: The final stop event containing the answer.
            
        Raises:
            RetrievalException: If answer generation fails.
        """
        if not ev.nodes:
            return StopEvent(result={"answer": "I do not have enough context to answer this query.", "source_nodes": [], "from_cache": False})

        self.send_event(StreamingStatusEvent(status="Reranking context..."))
        
        try:
            reranked_nodes = self.reranker.postprocess_nodes(ev.nodes, query_bundle=ev.query_bundle)
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error reranking: {e}", exc_info=True)
            reranked_nodes = ev.nodes
        
        self.send_event(StreamingStatusEvent(status="Reordering context (Lost in the Middle)..."))
        try:
            final_nodes = self.reorder.postprocess_nodes(reranked_nodes)
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error reordering: {e}", exc_info=True)
            final_nodes = reranked_nodes
        
        self.send_event(StreamingStatusEvent(status="Generating final answer..."))
        context_str = "\n".join([n.get_content() for n in final_nodes])
        final_prompt = (
            f"Context:\n{context_str}\n\n"
            f"Question: {ev.query_bundle.query_str}\n\n"
            f"Answer the question based on the context provided."
        )
        
        try:
            response = await self._call_llm_with_retry(final_prompt)
            answer = response.text
            self.cache.set_cache(ev.query_bundle.query_str, answer)
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error generating final answer: {e}", exc_info=True)
            raise RetrievalException(f"Failed to generate answer: {e}")
        
        return StopEvent(result={
            "answer": answer,
            "source_nodes": final_nodes,
            "from_cache": False
        })
