# ADR 002: LlamaIndex Workflows for Event-Driven RAG

## Status
Accepted

## Context
RAG (Retrieval-Augmented Generation) pipelines are often linear and brittle. As we introduce complex logic—such as query decomposition, HyDE (Hypothetical Document Embeddings), and relevance judgments with CoT (Chain of Thought)—a standard linear pipeline becomes difficult to maintain and scale. We need an architecture that supports loops, conditional branching, and streaming.

## Decision
We decided to adopt **LlamaIndex Workflows (Event-Driven API)** over standard QueryPipelines or LangChain's LangGraph. 

## Rationale
- **Event-Driven Nature:** Workflows in LlamaIndex allow steps to emit and consume typed events, making it trivial to add new steps (e.g., a PII masker or a Cache layer) without breaking the entire chain.
- **Cycles and Refinement:** The event-driven model natively supports loops. If a context is judged irrelevant, the workflow simply emits a `QueryTransformedEvent` again to refine the query until a limit is reached.
- **Streaming:** It provides native streaming of both status updates and final tokens, which is crucial for perceived user latency.
- **Pythonic:** Uses native Python `async/await` and Pydantic validation, aligning with our strict typing standards.

## Consequences
- **Positive:** High maintainability, easy to add observability per step, and excellent resilience.
- **Negative:** Slightly steeper learning curve for developers accustomed to simple linear chains.