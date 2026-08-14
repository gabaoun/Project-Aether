"""Lazy re-export: importing this package must not pull in torch/transformers.

AetherLoRAReranker is unused by the live pipeline (src/pipeline/retrieval.py
uses llama_index's own FlagEmbeddingReranker, gated behind enable_reranker).
An eager `from src.retrieval.reranker import AetherLoRAReranker` here loaded
torch+transformers on every import of this package, including at app startup,
regardless of the enable_reranker flag - the actual cause of the Render OOM
kill (exit 137). Tests import it directly from src.retrieval.reranker, which
still works unchanged.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.reranker import AetherLoRAReranker

__all__ = ["AetherLoRAReranker"]


def __getattr__(name: str):
    if name == "AetherLoRAReranker":
        from src.retrieval.reranker import AetherLoRAReranker

        return AetherLoRAReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
