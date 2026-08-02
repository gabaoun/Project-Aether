
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for Project Aether.
    """
    openai_api_key: str | None = None
    # Only used by scripts/migrate_to_chroma.py (the one-time Qdrant -> Chroma
    # Cloud migration) - the live app runs on Chroma exclusively, see below.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "project_aether_docs"

    chroma_host: str = "api.trychroma.com"
    chroma_api_key: str | None = None
    chroma_tenant: str = "d229b721-6e42-4d8a-800d-54f2d56651a6"
    chroma_database: str = "RAGabaoun"
    chroma_collection: str = "project_aether_docs"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    semantic_cache_threshold: float = 0.85
    
    database_url: str = "postgresql://user:password@postgres:5432/aether"
    
    phoenix_collector_endpoint: str = "http://localhost:6006"
    log_level: str = "INFO"
    
    data_dir: str = "./data"
    debug: bool = False
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
