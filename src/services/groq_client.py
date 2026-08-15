"""
Thin async wrapper around the official `groq` SDK, exposing just the
`.acomplete(prompt) -> object with .text` surface that RetrievalWorkflow
(src/pipeline/retrieval.py) and the ingestion metadata-enrichment step
(src/pipeline/ingestion.py) actually use.

Exists because `llama_index.llms.groq.Groq` - despite the name - inherits
from `llama_index.llms.openai_like`, which unconditionally imports
`transformers` (pulling in torch, sklearn, scipy, pandas - ~470-510MB RSS,
measured 2026-08-14) even though nothing in this app's Groq usage needs any
of that. The raw `groq` SDK alone costs ~34MB. On Render's 512MB free tier
(render.yaml, `plan: free`) that difference is the whole ballgame - this
swap is what keeps the *always-loaded* /query path (not the opt-in reranker
or LangChain engine) inside budget.

Not a drop-in llama_index BaseLLM - src/services/neo4j.py's PropertyGraphIndex
usage genuinely needs that full interface, so it keeps importing the real
`llama_index.llms.groq.Groq` (lazily, only when ENABLE_NEO4J is on).
"""

from dataclasses import dataclass

from groq import AsyncGroq


@dataclass
class _CompletionResult:
    text: str


class LightweightGroqLLM:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self._client = AsyncGroq(api_key=api_key)

    async def acomplete(self, prompt: str) -> _CompletionResult:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return _CompletionResult(text=text)
