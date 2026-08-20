
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for Project Aether.
    """
    groq_api_key: str | None = None
    # Only used by scripts/migrate_to_chroma.py (the one-time Qdrant -> Chroma
    # Cloud migration) - the live app runs on Chroma exclusively, see below.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "project_aether_docs"

    chroma_api_key: str | None = None
    chroma_tenant: str = "d229b721-6e42-4d8a-800d-54f2d56651a6"
    chroma_database: str = "RAGabaoun"
    chroma_collection: str = "project_aether_docs"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_ssl: bool = False
    semantic_cache_threshold: float = 0.85
    
    database_url: str = "postgresql://user:password@postgres:5432/aether"
    
    phoenix_collector_endpoint: str = "http://localhost:6006"
    log_level: str = "INFO"

    # BAAI/bge-reranker-v2-m3 is ~600M params (~2GB+ resident) - loading it
    # eagerly at startup OOM-kills a 512MB free-tier host before the port
    # ever opens. Off by default; enable on a host with enough headroom.
    enable_reranker: bool = False

    # langchain-core alone costs ~230-290MB RSS just to import (measured
    # 2026-08-14, python -c "from langchain_core... import ...") - close
    # to the entire 512MB budget on Render's free tier by itself, before
    # the rest of the app (llama-index-core, chromadb, fastapi) is even
    # loaded. Off by default for the same reason as enable_reranker above;
    # src/api/app.py keeps the import itself inside the flag, not just the
    # instantiation, so disabling this actually avoids paying the cost.
    enable_langchain_engine: bool = False

    enable_neo4j: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    data_dir: str = "./data"
    debug: bool = False

    portfolio_mode: bool = False

    # Origins allowed to call the API from a browser (CORS). Comma-separated
    # via env var CORS_ORIGINS. Defaults cover the portfolio site + local dev.
    cors_origins: str = "https://gabaoun.github.io,http://localhost:3000,http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Shared secret required (via X-Admin-Token header) to hit POST /ingest.
    # If unset, /ingest stays open - fine for local dev, not for a public
    # deploy (it enqueues a full Postgres/RQ job + Chroma collection rebuild).
    admin_token: str | None = None
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
