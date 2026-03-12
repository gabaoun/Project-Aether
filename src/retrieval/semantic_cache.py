import json
import numpy as np
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from typing import Optional, Dict, Union, List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.utils.config import settings
from src.core.exceptions import CacheException
from src.utils.logger import logger

class SemanticCache:
    """
    A semantic caching layer utilizing Redis and Vector Similarity Search
    to reduce LLM latency and API costs.
    
    Attributes:
        index_name (str): The name of the Redis index.
        prefix (str): Prefix for cache keys.
        vector_dim (int): The dimensionality of the embeddings.
        enabled (bool): Whether the Redis connection is active.
    """
    def __init__(self) -> None:
        """Initializes the Semantic Cache by connecting to Redis and setting up the HNSW index."""
        self.index_name = "cache_index"
        self.prefix = "cache:"
        self.vector_dim = 384 # BGE-small embedding dimension
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host, 
                port=settings.redis_port, 
                decode_responses=False,
                socket_timeout=2
            )
            self.redis_client.ping()
            self.enabled = True
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._setup_index()
            logger.info("[CACHE_LAYER] Redis Semantic Cache initialized with Vector Search.")
        except Exception as e:
            logger.warning(f"[CACHE_LAYER] Redis not available, Semantic Cache disabled: {e}")
            self.enabled = False

    def _setup_index(self) -> None:
        """Sets up the RediSearch HNSW index if it does not already exist."""
        try:
            self.redis_client.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            schema = (
                TextField("query"),
                TextField("answer"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )
            definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
            self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
            logger.info(f"[CACHE_LAYER] Created RediSearch HNSW index '{self.index_name}'.")

    def get_cache(self, query: str) -> Optional[str]:
        """
        Retrieves a cached answer if a semantically similar query exists.
        
        Args:
            query (str): The user query.
            
        Returns:
            Optional[str]: The cached answer if found and similarity threshold met, else None.
        """
        if not self.enabled:
            return None
        
        try:
            query_embedding = self.embed_model.get_text_embedding(query)
            embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            q = Query(f"*=>[KNN 1 @embedding $query_vector AS score]")\
                .sort_by("score")\
                .return_fields("answer", "score")\
                .dialect(2)
            
            res = self.redis_client.ft(self.index_name).search(q, query_params={"query_vector": embedding_bytes})
            
            if res.docs:
                doc = res.docs[0]
                distance = float(doc.score)
                similarity = 1.0 - distance
                
                if similarity > settings.semantic_cache_threshold:
                    logger.info(f"[CACHE_LAYER] Semantic Cache HIT! Similarity: {similarity:.4f}")
                    return doc.answer.decode("utf-8") if isinstance(doc.answer, bytes) else doc.answer
            
            logger.info("[CACHE_LAYER] Semantic Cache MISS.")
            return None
        except Exception as e:
            logger.error(f"[CACHE_LAYER] Error reading cache: {e}", exc_info=True)
            return None

    def set_cache(self, query: str, answer: str) -> None:
        """
        Caches a query and its corresponding answer.
        
        Args:
            query (str): The user query.
            answer (str): The generated answer to cache.
        """
        if not self.enabled:
            return
        try:
            query_embedding = self.embed_model.get_text_embedding(query)
            embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            cache_id = f"{self.prefix}{abs(hash(query))}"
            
            self.redis_client.hset(
                cache_id,
                mapping={
                    "query": query,
                    "embedding": embedding_bytes,
                    "answer": answer
                }
            )
            self.redis_client.expire(cache_id, 3600)
        except Exception as e:
            logger.error(f"[CACHE_LAYER] Error setting cache: {e}", exc_info=True)

    def invalidate_cache(self) -> None:
        """
        Drops and recreates the cache index to invalidate all cached items.
        
        Raises:
            CacheException: If the invalidation process fails.
        """
        if not self.enabled:
            return
        try:
            self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
            self._setup_index()
            logger.info("[CACHE_LAYER] Cache invalidated successfully.")
        except Exception as e:
            logger.error(f"[CACHE_LAYER] Error invalidating cache: {e}", exc_info=True)
            raise CacheException(f"Failed to invalidate cache: {e}")
