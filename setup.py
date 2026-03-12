from setuptools import setup, find_packages

setup(
    name="project-aether",
    version="0.1.0",
    author="Gabriel (Gabaoun) Penha",
    description="Advanced RAG Engine with Event-Driven Workflows",
    packages=find_packages(),
    install_requires=[
        "llama-index>=0.11.0",
        "llama-index-core",
        "llama-index-readers-file",
        "llama-index-embeddings-huggingface",
        "llama-index-llms-openai",
        "llama-index-postprocessor-flag-embedding-reranker",
        "llama-index-vector-stores-qdrant",
        "llama-index-vector-stores-postgres",
        "pydantic>=2.0",
        "pydantic-settings",
        "qdrant-client",
        "psycopg2-binary",
        "ragas",
        "python-dotenv",
        "arize-phoenix",
        "llama-index-utils-workflow",
        "sentence-transformers",
        "FlagEmbedding",
    ],
)
