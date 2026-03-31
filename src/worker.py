import redis
from rq import Worker, Queue, Connection
from src.config.settings import settings
from src.utils.logger import logger

redis_conn = redis.Redis(host=settings.redis_host, port=settings.redis_port)
queue = Queue("default", connection=redis_conn)

def run_worker():
    logger.info(f"Starting worker connected to Redis at {settings.redis_host}:{settings.redis_port}")
    with Connection(redis_conn):
        worker = Worker([queue])
        worker.work()

if __name__ == "__main__":
    run_worker()
