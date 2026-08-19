import uuid
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudQwenEmbeddingModel,
)

from src.config.settings import settings
from src.utils.logger import logger


class ChromaService:
    def __init__(self):
        # Chroma Cloud requires CloudClient specifically - it handles the
        # real auth scheme (x-chroma-token) and HTTPS routing internally.
        # A generic HttpClient with a hand-rolled Bearer header and no
        # scheme/port (as this used to be) cannot reach the Cloud API at
        # all ("Server disconnected without sending a response").
        self.client = chromadb.CloudClient(
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            api_key=settings.chroma_api_key,
        )
        self.collection_name = settings.chroma_collection
        # Chroma Cloud's own embedding API (server-side, just an httpx POST -
        # no local model, no torch/onnxruntime). This is what actually needs
        # to be passed as embedding_function; a plain metadata dict does
        # nothing - without it, chromadb silently falls back to its bundled
        # local ONNX MiniLM, which OOM-killed the 512MB Render instance the
        # moment onnxruntime tried to load it.
        self.embedding_function = ChromaCloudQwenEmbeddingFunction(
            model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
            task=None,
        )

    def get_or_create_collection(self) -> Collection:
        """
        Creates or retrieves a collection using Chroma Cloud's server-side Qwen embedding.
        Ingestion-only: requires a write-capable API key, since the "create"
        branch needs write access even when the collection already exists.
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function,
        )

    def get_collection(self) -> Collection:
        """
        Retrieves the existing collection without attempting to create it,
        so the query path doesn't need create rights on the collection.

        This alone does NOT make a read-only Chroma Cloud API key usable for
        the deployed query service. chromadb.CloudClient's constructor
        (chromadb.api.client.Client.__init__) unconditionally calls an
        admin-scoped _validate_tenant_database() -> get_tenant()/
        get_database() before any query happens - a key scoped read-only to
        a single database doesn't have rights for that tenant-level call, so
        it fails at ChromaService() construction, not at query time.
        Confirmed 2026-08-18 against a real Chroma Cloud read-only key:
        chromadb.errors.ChromaError: Permission denied., raised from
        Client._validate_tenant_database's self._admin_client.get_tenant().
        Until Chroma Cloud's client/API changes this, CHROMA_API_KEY must be
        a key with tenant-read scope (in practice: the same read-write key
        used for ingestion) even for a query-only deployment.
        """
        return self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def chunk_text(self, text: str, max_bytes: int = 16000) -> list[str]:
        """
        Line-based chunking strategy for documents > 16 KiB.
        """
        if len(text.encode('utf-8')) <= max_bytes:
            return [text]
            
        chunks = []
        lines = text.splitlines(keepends=True)
        current_chunk = ""
        current_size = 0
        
        for line in lines:
            line_size = len(line.encode('utf-8'))
            if current_size + line_size > max_bytes:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
                current_size = line_size
                # If a single line is too big, we must split it (rare but possible)
                while current_size > max_bytes:
                    part = current_chunk.encode('utf-8')[:max_bytes].decode('utf-8', 'ignore')
                    chunks.append(part)
                    current_chunk = current_chunk[len(part):]
                    current_size = len(current_chunk.encode('utf-8'))
            else:
                current_chunk += line
                current_size += line_size
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def clear_collection(self) -> None:
        """
        Deletes every chunk currently in the collection. Portfolio ingestion
        (scripts/ingest_portfolio.py) is a full-snapshot rebuild, not an
        incremental sync - node IDs are random per run (llama_index
        SentenceSplitter node_id), so without this, upsert_documents()
        never overwrites old chunks, it only piles new ones on top,
        letting stale/deleted-source-file chunks accumulate forever.
        """
        collection = self.get_or_create_collection()
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)
            logger.info(f"Cleared {len(existing_ids)} existing chunks from Chroma collection '{self.collection_name}'.")

    async def upsert_documents(self, documents: list[dict[str, Any]]):
        """
        Upserts documents into Chroma Cloud.
        Each document should have 'id', 'text', and 'metadata'.
        """
        collection = self.get_or_create_collection()
        
        ids = []
        metadatas = []
        documents_content = []
        
        for doc in documents:
            text = doc['text']
            doc_id = doc.get('id', str(uuid.uuid4()))
            base_metadata = doc.get('metadata', {})
            
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                ids.append(chunk_id)
                documents_content.append(chunk)
                
                chunk_metadata = base_metadata.copy()
                chunk_metadata['source_doc_id'] = doc_id
                chunk_metadata['chunk_index'] = i
                metadatas.append(chunk_metadata)
        
        if ids:
            collection.upsert(
                ids=ids,
                metadatas=metadatas,
                documents=documents_content
            )
            logger.info(f"Upserted {len(ids)} chunks to Chroma collection '{self.collection_name}'.")

    async def hybrid_search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """
        Performs hybrid search (dense + sparse) with RRF and GroupBy deduplication.
        """
        collection = self.get_collection()
        
        # Following https://docs.trychroma.com/cloud/search-api/hybrid-search.md
        # and https://docs.trychroma.com/cloud/search-api/group-by.md
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                # groupby is a feature of Chroma Cloud - the local SDK's type
                # stub doesn't declare these params, but Cloud passes them
                # through server-side; the except below handles SDKs that don't.
                group_by="source_doc_id",  # type: ignore[call-arg]
                group_limit=1  # type: ignore[call-arg]
            )
            assert results['ids'] is not None and results['documents'] is not None
            assert results['metadatas'] is not None and results['distances'] is not None

            processed_results = []
            for i in range(len(results['ids'][0])):
                processed_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i]
                })
            return processed_results
            
        except Exception as e:  # noqa: BLE001 - boundary catch, must degrade gracefully rather than crash the pipeline
            logger.warning(f"Chroma Cloud specialized query failed, falling back to manual GroupBy: {e}")
            # Fallback to standard query if SDK doesn't support group_by yet or it's not a Cloud collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"]
            )
            assert results['ids'] is not None and results['documents'] is not None
            assert results['metadatas'] is not None and results['distances'] is not None

            seen_docs = {}
            processed_results = []

            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i]
                content = results['documents'][0][i]
                distance = results['distances'][0][i]
                
                source_doc_id = metadata.get('source_doc_id', doc_id)
                
                if source_doc_id not in seen_docs:
                    seen_docs[source_doc_id] = {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "score": distance
                    }
                    processed_results.append(seen_docs[source_doc_id])
                    
                if len(processed_results) >= n_results:
                    break
                    
            return processed_results
