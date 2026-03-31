# ADR 001: Simplified Semantic Caching with Redis

## Status
Accepted

## Context
RAG systems can be slow and expensive if the same queries are sent to the LLM repeatedly. We need a way to cache responses for identical or highly similar queries.

## Decision
We decided to implement a simplified caching layer using Redis as a key-value store.

## Rationale
- **Speed:** Redis provides sub-millisecond lookups for cached answers.
- **Cost Reduction:** Intercepting recurrent queries prevents redundant LLM API calls.
- **Implementation Clarity:** For the current project scope, a key-based lookup (using query strings) provides immediate value without the overhead of managing a separate vector index within Redis.

## Consequences
- **Positive:** Immediate performance gain for repeated queries and reduced costs.
- **Negative:** Current implementation requires exact query matching (or highly similar pre-processed strings) and does not yet leverage true vector-based similarity search in Redis.

## Future Work
In a production-grade system, this could be migrated to utilize Redis Stack's native vector search capabilities (HNSW) to allow for a true semantic cache.
