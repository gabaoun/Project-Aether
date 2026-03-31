import redis
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config.settings import settings
from src.utils.logger import logger

class SemanticCache:
    """
    Redis-based semantic cache for LLM responses.
    """
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True
        )
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.threshold = settings.semantic_cache_threshold

    def _get_embedding(self, text: str):
        return self.embed_model.get_text_embedding(text)

    def get_cache(self, query: str):
        try:
            # For simplicity, we use exact match in this implementation.
            # A true semantic cache would use vector search in Redis.
            cached_res = self.redis_client.get(f"cache:{query}")
            if cached_res:
                return cached_res
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None

    def set_cache(self, query: str, answer: str):
        try:
            self.redis_client.setex(f"cache:{query}", 3600, answer)
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    def invalidate_cache(self):
        try:
            keys = self.redis_client.keys("cache:*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
        logger.info("Semantic cache invalidated.")
