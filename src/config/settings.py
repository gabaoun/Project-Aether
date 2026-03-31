from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Configuration settings for Project Aether.
    """
    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "project_aether_docs"

    chroma_host: str = "api.trychroma.com"
    chroma_api_key: Optional[str] = None
    chroma_tenant: str = "d229b721-6e42-4d8a-800d-54f2d56651a6"
    chroma_database: str = "RAGabaoun"
    chroma_collection: str = "project_aether_docs"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    semantic_cache_threshold: float = 0.85
    
    phoenix_collector_endpoint: str = "http://localhost:6006"
    log_level: str = "INFO"
    
    data_dir: str = "./data"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
