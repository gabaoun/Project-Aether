import asyncio
from typing import Any

import torch
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import PrivateAttr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logger import logger


class AetherLoRAReranker(BaseNodePostprocessor):
    base_model_name: str = "BAAI/bge-reranker-v2-m3"
    adapter_name_or_path: str | None = None
    top_n: int = 5
    max_length: int = 512
    fuse_adapter: bool = True

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        base_model_name: str = "BAAI/bge-reranker-v2-m3",
        adapter_name_or_path: str | None = None,
        top_n: int = 5,
        max_length: int = 512,
        fuse_adapter: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.adapter_name_or_path = adapter_name_or_path
        self.top_n = top_n
        self.max_length = max_length
        self.fuse_adapter = fuse_adapter
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model()

    def _init_model(self) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_name,
                num_labels=1,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )

            if self.adapter_name_or_path:
                from peft import PeftModel

                peft_model = PeftModel.from_pretrained(model, self.adapter_name_or_path)
                if self.fuse_adapter:
                    model = peft_model.merge_and_unload()
                else:
                    model = peft_model

            model.to(self._device)
            model.eval()
            self._model = model
            logger.info(f"[RERANKER] Successfully initialized reranker with base={self.base_model_name}, adapter={self.adapter_name_or_path}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[RERANKER] Failed to initialize reranker model: {e}")
            self._model = None
            self._tokenizer = None

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or not query_bundle or not query_bundle.query_str:
            return nodes[: self.top_n]

        if self._model is None or self._tokenizer is None:
            logger.warning("[RERANKER] Model/Tokenizer not loaded. Returning un-reranked nodes.")
            return nodes[: self.top_n]

        try:
            query = query_bundle.query_str
            passages = [node.get_content() for node in nodes]
            pairs = [[query, passage] for passage in passages]

            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._device)
            else:
                inputs = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

            with torch.no_grad():
                logits = self._model(**inputs).logits.squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                scores = torch.sigmoid(logits).cpu().tolist()

            reranked_nodes: list[NodeWithScore] = []
            for node, score in zip(nodes, scores):
                reranked_nodes.append(
                    NodeWithScore(node=node.node, score=float(score))
                )

            reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            return reranked_nodes[: self.top_n]
        except Exception as e:  # noqa: BLE001
            logger.error(f"[RERANKER] Error during node postprocessing: {e}")
            return nodes[: self.top_n]

    async def apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        return await asyncio.to_thread(self._postprocess_nodes, nodes, query_bundle)
