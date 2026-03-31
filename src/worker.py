from rq import Worker
from src.infra.queue import get_queue, get_redis_connection
from src.utils.logger import logger

def run_worker():
    redis_conn = get_redis_connection()
    queue = get_queue()
    
    logger.info("Starting worker connected to Redis")
    
    # In recent RQ versions, connection is passed to the Worker constructor
    worker = Worker([queue], connection=redis_conn)
    worker.work()

if __name__ == "__main__":
    run_worker()
