# ADR 001: Migration to Native Vector Search in Redis for Semantic Caching

## Context
The previous version of Project Aether implemented a semantic cache layer (`SemanticCache`) using Redis purely as a key-value store. It fetched all cache keys via an O(N) `keys("cache:*")` scan, extracted embeddings, and computed cosine similarity locally in Python using `numpy`. This architecture created a significant bottleneck:
- **O(N) Complexity:** As the cache grew, response times degraded linearly.
- **GIL Contention:** Numpy calculations, while fast in C, still required moving large amounts of data into Python's memory space, blocking the Global Interpreter Lock (GIL) and reducing overall system throughput.
- **Network Overhead:** Fetching hundreds or thousands of JSON payloads over the network for every query was highly inefficient.

## Decision
We are migrating the `SemanticCache` module to utilize **RediSearch and RedisJSON (Redis Stack)**. We will define an **HNSW (Hierarchical Navigable Small World)** vector index directly within Redis.
1. Documents will be stored in Redis as Hashes (or JSON).
2. The index will track the `embedding` vector field using the `HNSW` algorithm with `COSINE` distance.
3. Queries will utilize the `FT.SEARCH` command with `KNN` syntax to let the Redis engine natively compute and return the top match in sub-milliseconds.

## Consequences
- **Positive:**
  - **Latency:** Cache hits are now resolved in milliseconds regardless of the cache size.
  - **Throughput:** Computation is offloaded from the Python application to the Redis C-engine, avoiding GIL blocking.
  - **Network:** Only the best matching result (or a small set) is returned over the network, minimizing bandwidth.
- **Negative/Requirements:**
  - Requires the deployment of **Redis Stack** (which includes RediSearch) rather than a vanilla Redis instance.
  - Requires explicit index management (creation on startup) and memory tuning for the HNSW index in Redis.

## Status
Accepted
