from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Configuration settings for Project Aether.
    
    Attributes:
        openai_api_key (str): The API key for OpenAI services.
        qdrant_url (str): The URL for the Qdrant vector database.
        qdrant_api_key (Optional[str]): The API key for Qdrant (if applicable).
        postgres_user (str): PostgreSQL database username.
        postgres_password (str): PostgreSQL database password.
        postgres_db (str): PostgreSQL database name.
        postgres_host (str): PostgreSQL database host.
        postgres_port (int): PostgreSQL database port.
        redis_host (str): Redis host for semantic caching.
        redis_port (int): Redis port.
        semantic_cache_threshold (float): Similarity threshold for cache hits.
        phoenix_collector_endpoint (str): Endpoint for Arize Phoenix observability.
        log_level (str): Logging level.
        data_dir (str): Directory for local data ingestion.
    """
    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "project_aether"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    semantic_cache_threshold: float = 0.85
    
    phoenix_collector_endpoint: str = "http://localhost:6006"
    log_level: str = "INFO"
    
    data_dir: str = "./data"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
