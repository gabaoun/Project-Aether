from collections.abc import Sequence
from typing import Any

from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    LLMSynonymRetriever,
    PropertyGraphIndex,
    SimpleLLMPathExtractor,
)
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.llms.groq import Groq

from src.config.settings import settings
from src.utils.logger import logger


class Neo4jService:
    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self.database = database or settings.neo4j_database
        self.graph_store: Any | None = None

        if settings.enable_neo4j:
            try:
                from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

                self.graph_store = Neo4jPropertyGraphStore(
                    username=self.username,
                    password=self.password,
                    url=self.uri,
                    database=self.database,
                )
                logger.info(f"[NEO4J] Successfully connected to Neo4j at {self.uri}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[NEO4J] Failed to initialize Neo4jPropertyGraphStore: {e}")
                self.graph_store = None
        else:
            logger.info("[NEO4J] Neo4j is disabled (ENABLE_NEO4J=False).")

    def is_enabled(self) -> bool:
        return bool(settings.enable_neo4j and self.graph_store is not None)

    def get_graph_store(self) -> Any | None:
        return self.graph_store if self.is_enabled() else None

    async def extract_and_index_nodes(
        self, nodes: Sequence[BaseNode], llm: Any = None
    ) -> None:
        if not self.is_enabled() or not nodes:
            return

        try:
            extractor_llm = llm or self._get_llm()
            kg_extractors: list[Any] = [ImplicitPathExtractor()]
            if extractor_llm:
                kg_extractors.insert(0, SimpleLLMPathExtractor(llm=extractor_llm))

            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                llm=extractor_llm,
                embed_model=None,
                embed_kg_nodes=False,
            )
            await index.ainsert_nodes(nodes, kg_extractors=kg_extractors)
            logger.info(f"[NEO4J] Extracted entities and indexed {len(nodes)} nodes into Neo4j graph store.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[NEO4J] Error extracting and indexing nodes: {e}")

    async def retrieve(
        self, query_str: str, llm: Any = None
    ) -> list[NodeWithScore]:
        if not self.is_enabled() or not query_str:
            return []

        try:
            retriever_llm = llm or self._get_llm()
            index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                llm=retriever_llm,
                embed_model=None,
                embed_kg_nodes=False,
            )
            sub_retriever = LLMSynonymRetriever(
                graph_store=self.graph_store, llm=retriever_llm
            )
            retriever = index.as_retriever(sub_retrievers=[sub_retriever])
            nodes = await retriever.aretrieve(query_str)
            logger.info(f"[NEO4J] Retrieved {len(nodes)} nodes from Neo4j.")
            return list(nodes)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[NEO4J] Error retrieving graph context: {e}")
            return []

    def _get_llm(self) -> Any | None:
        if settings.groq_api_key:
            return Groq(model="llama-3.3-70b-versatile", api_key=settings.groq_api_key)
        return None

    def close(self) -> None:
        if self.graph_store and hasattr(self.graph_store, "close"):
            try:
                self.graph_store.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Error closing graph store: {e}")
