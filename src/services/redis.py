import redis

from src.config.settings import settings
from src.utils.logger import logger


class SemanticCache:
    """
    Redis-based semantic cache for LLM responses.
    Handles connection failures gracefully (degraded mode).
    """
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                ssl=settings.redis_ssl,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            self.enabled = True
        except (redis.ConnectionError, redis.TimeoutError):
            logger.warning("Redis not available. Semantic cache disabled (degraded mode).")
            self.enabled = False
            
        self.threshold = settings.semantic_cache_threshold

    def get_cache(self, query: str):
        if not self.enabled:
            return None
        try:
            return self.redis_client.get(f"cache:{query}")
        except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
            logger.error(f"Cache retrieval error: {e}")
            return None

    def set_cache(self, query: str, answer: str):
        if not self.enabled:
            return
        try:
            self.redis_client.setex(f"cache:{query}", 3600, answer)
        except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
            logger.error(f"Cache storage error: {e}")

    def invalidate_cache(self):
        if not self.enabled:
            return
        try:
            keys = self.redis_client.keys("cache:*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info("Semantic cache invalidated.")
        except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
            logger.error(f"Cache invalidation error: {e}")
