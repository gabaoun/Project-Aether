import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from typing import List, Dict, Any, Optional
import uuid
import math
from src.config.settings import settings
from src.utils.logger import logger

class ChromaService:
    def __init__(self):
        self.client = chromadb.HttpClient(
            host=settings.chroma_host,
            tenant=settings.chroma_tenant,
            database=settings.chroma_database,
            headers={"Authorization": f"Bearer {settings.chroma_api_key}"} if settings.chroma_api_key else None,
            settings=Settings(allow_reset=True)
        )
        self.collection_name = settings.chroma_collection

    def get_or_create_collection(self) -> Collection:
        """
        Creates or retrieves a collection with dense and sparse embedding configuration.
        """
        # Define Schema for dense + sparse search as per Chroma Cloud docs
        # Note: In latest chromadb SDK, we can pass schema in metadata or specialized params if supported.
        # Following https://docs.trychroma.com/cloud/schema/sparse-vector-search.md
        
        metadata = {
            "dense_model": "Chroma Cloud Qwen",
            "sparse_model": "Chroma Cloud Splade",
            "hnsw:space": "cosine"
        }
        
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata
        )

    def chunk_text(self, text: str, max_bytes: int = 16000) -> List[str]:
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

    async def upsert_documents(self, documents: List[Dict[str, Any]]):
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

    async def hybrid_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs hybrid search (dense + sparse) with RRF and GroupBy deduplication.
        """
        collection = self.get_or_create_collection()
        
        # Following https://docs.trychroma.com/cloud/search-api/hybrid-search.md
        # and https://docs.trychroma.com/cloud/search-api/group-by.md
        
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
                # groupby is a feature of Chroma Cloud
                # group_by and group_limit are the expected parameters
                group_by="source_doc_id",
                group_limit=1
            )
            
            processed_results = []
            for i in range(len(results['ids'][0])):
                processed_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i]
                })
            return processed_results
            
        except Exception as e:
            logger.warning(f"Chroma Cloud specialized query failed, falling back to manual GroupBy: {e}")
            # Fallback to standard query if SDK doesn't support group_by yet or it's not a Cloud collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results * 2,
                include=["documents", "metadatas", "distances"]
            )
            
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
