# ADR 003: Semantic Chunking Strategy

## Status
Accepted

## Context
Standard text splitters (like recursive character splitters) often cut documents in the middle of a thought, degrading the quality of the embeddings and the subsequent retrieval. Furthermore, processing thousands of documents into semantic chunks can lead to Out-Of-Memory (OOM) errors if all chunks are held in memory simultaneously.

## Decision
We decided to implement a custom **SemanticDoubleMergingSplitter** using Python Generators.

## Rationale
- **Semantic Coherence:** By using embeddings to find semantic breakpoints in the text, we ensure that chunks contain complete thoughts.
- **Double Merging:** We added a second pass to merge chunks that are too small (below `min_chunk_size`). Small chunks often lack enough context for the LLM to generate a good answer, even if semantically distinct.
- **Memory Efficiency (Generators):** Instead of processing all documents and returning a massive list of nodes, the splitter yields nodes one-by-one via `get_nodes_generator()`. This keeps the memory footprint low and constant, allowing us to ingest gigabytes of text on standard hardware.

## Consequences
- **Positive:** Higher retrieval accuracy and prevention of memory bloat during the ingestion phase.
- **Negative:** Slower ingestion time compared to naive character splitting, as semantic splitting requires computing embeddings for sentences to find breakpoints.