from typing import List, Union
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
    Context,
)
from llama_index.core.postprocessor import LongContextReorder
from llama_index.llms.openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.services.chroma import ChromaService
from src.services.redis import SemanticCache
from src.utils.token_counter import TokenCounter
from src.utils.logger import logger
from src.models.exceptions import RetrievalException

class StreamingStatusEvent(Event):
    status: str

class QueryTransformedEvent(Event):
    query_bundle: QueryBundle
    loops: int

class ContextRetrievedEvent(Event):
    nodes: List[NodeWithScore]
    query_bundle: QueryBundle
    loops: int

class RelevanceJudgedEvent(Event):
    is_relevant: bool
    nodes: List[NodeWithScore]
    query_bundle: QueryBundle

class RetrievalWorkflow(Workflow):
    """
    Workflow for RAG retrieval with query transformation, reranking, and cache checks using Chroma Cloud.      
    """
    def __init__(self, chroma_service: ChromaService, reranker=None, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.chroma_service = chroma_service
        self.llm = OpenAI(model="gpt-4o", api_key=settings.openai_api_key)
        self.reranker = reranker or self._build_reranker()
        self.reorder = LongContextReorder()
        self.cache = SemanticCache()
        self.token_counter = TokenCounter(model_name="gpt-4o")

    def _build_reranker(self):
        try:
            from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
            return FlagEmbeddingReranker(
                model="BAAI/bge-reranker-v2-m3",
                top_n=5
            )
        except ImportError:
            logger.warning("FlagEmbeddingReranker not available. Skipping reranking.")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm_with_retry(self, prompt: str):
        response = await self.llm.acomplete(prompt)
        self.token_counter.log_cost("RetrievalLLM", prompt, response.text)
        return response

    @step
    async def process_start(self, ctx: Context, ev: StartEvent) -> Union[QueryTransformedEvent, StopEvent, StreamingStatusEvent]:
        query_str = ev.get("query")
        if not query_str:
            raise RetrievalException("query must be provided in StartEvent", status_code=400)

        # 1. Cache Check
        cached_answer = self.cache.get_cache(query_str)
        if cached_answer:
            ctx.send_event(StreamingStatusEvent(status="Cache Hit! Returning cached response."))
            return StopEvent(result={"answer": cached_answer, "source_nodes": [], "from_cache": True})

        ctx.send_event(StreamingStatusEvent(status="Transforming query..."))

        # Decompose & HyDE (Simplified)
        hyde_prompt = f"Write a hypothetical document that would answer the following question: {query_str}"
        try:
            hyde_doc = await self._call_llm_with_retry(hyde_prompt)
            custom_embeddings = [query_str, hyde_doc.text]
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error generating HyDE: {e}")
            custom_embeddings = [query_str]
            
        query_bundle = QueryBundle(query_str=query_str, custom_embedding_strs=custom_embeddings)
        return QueryTransformedEvent(query_bundle=query_bundle, loops=0)

    @step
    async def retrieve_context(self, ctx: Context, ev: QueryTransformedEvent) -> Union[ContextRetrievedEvent, StreamingStatusEvent]:
        ctx.send_event(StreamingStatusEvent(status="Retrieving context from Chroma Cloud..."))
        
        # Using Chroma Cloud Hybrid Search
        results = await self.chroma_service.hybrid_search(ev.query_bundle.query_str, n_results=20)
        
        from llama_index.core.schema import TextNode
        nodes = []
        for res in results:
            node = NodeWithScore(
                node=TextNode(
                    text=res['content'],
                    id_=res['id'],
                    metadata=res['metadata']
                ),
                score=res['score']
            )
            nodes.append(node)

        return ContextRetrievedEvent(nodes=nodes, query_bundle=ev.query_bundle, loops=ev.loops)

    @step
    async def judge_relevance(self, ctx: Context, ev: ContextRetrievedEvent) -> Union[RelevanceJudgedEvent, QueryTransformedEvent, StreamingStatusEvent]:
        if ev.loops >= 1 or not ev.nodes:
            return RelevanceJudgedEvent(is_relevant=True, nodes=ev.nodes, query_bundle=ev.query_bundle)

        ctx.send_event(StreamingStatusEvent(status="Judging context relevance..."))
        context_text = "\n".join([n.get_content() for n in ev.nodes[:3]])
        
        judge_prompt = (
            f"Query: {ev.query_bundle.query_str}\n"
            f"Context:\n{context_text}\n"
            f"Does the context contain enough information to answer the query? VERDICT: YES or NO."
        )
        
        try:
            response = await self._call_llm_with_retry(judge_prompt)
            is_relevant = "YES" in response.text.upper()
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error judging relevance: {e}")
            is_relevant = True
        
        if not is_relevant:
            ctx.send_event(StreamingStatusEvent(status="Refining query..."))
            refine_prompt = f"Rewrite the query '{ev.query_bundle.query_str}' to be more specific for better search results."
            try:
                new_query_resp = await self._call_llm_with_retry(refine_prompt)
                new_bundle = QueryBundle(query_str=new_query_resp.text)
                return QueryTransformedEvent(query_bundle=new_bundle, loops=ev.loops + 1)
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error refining query: {e}")
            
        return RelevanceJudgedEvent(is_relevant=True, nodes=ev.nodes, query_bundle=ev.query_bundle)

    @step
    async def post_process(self, ctx: Context, ev: RelevanceJudgedEvent) -> Union[StopEvent, StreamingStatusEvent]:
        if not ev.nodes:
            return StopEvent(result={"answer": "No relevant context found.", "source_nodes": [], "from_cache": False})

        try:
            if self.reranker:
                ctx.send_event(StreamingStatusEvent(status="Reranking results..."))
                reranked_nodes = self.reranker.postprocess_nodes(ev.nodes, query_bundle=ev.query_bundle)
                final_nodes = self.reorder.postprocess_nodes(reranked_nodes)
            else:
                final_nodes = self.reorder.postprocess_nodes(ev.nodes)
        except Exception as e:
            logger.error(f"[RETRIEVAL] Post-processing error: {e}")
            final_nodes = ev.nodes
        
        ctx.send_event(StreamingStatusEvent(status="Generating answer..."))
        context_str = "\n".join([n.get_content() for n in final_nodes])
        final_prompt = f"Context:\n{context_str}\n\nQuestion: {ev.query_bundle.query_str}\n\nAnswer:"
        
        try:
            response = await self._call_llm_with_retry(final_prompt)
            answer = response.text
            self.cache.set_cache(ev.query_bundle.query_str, answer)
            return StopEvent(result={"answer": answer, "source_nodes": final_nodes, "from_cache": False})
        except Exception as e:
            logger.error(f"[RETRIEVAL] Answer generation failed: {e}")
            raise RetrievalException(f"Failed to generate answer: {e}")

