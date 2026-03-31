import redis
from rq import Queue
from src.config.settings import settings

# We don't initialize connection here directly to avoid side effects on import
# instead we provide a way to get the queue when needed

def get_redis_connection():
    return redis.Redis(host=settings.redis_host, port=settings.redis_port)

def get_queue(name="default"):
    return Queue(name, connection=get_redis_connection())
