from unittest.mock import MagicMock

import pytest
import torch
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval.reranker import AetherLoRAReranker


@pytest.fixture(autouse=True)
def mock_redis(mocker):
    import redis
    mock_r = MagicMock()
    mock_r.ping.side_effect = redis.ConnectionError("No Redis")
    mocker.patch("redis.Redis", return_value=mock_r)


@pytest.mark.asyncio
async def test_aether_lora_reranker_init_and_postprocess(mocker):
    mock_batch = MagicMock()
    mock_batch.items.return_value = [("input_ids", torch.tensor([[1, 2], [3, 4]]))]
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_batch
    mocker.patch(
        "src.retrieval.reranker.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )

    mock_model_output = MagicMock()
    mock_model_output.logits = torch.tensor([2.5, 0.1])
    mock_model = MagicMock()
    mock_model.return_value = mock_model_output
    mocker.patch(
        "src.retrieval.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_model,
    )

    reranker = AetherLoRAReranker(top_n=2)
    nodes = [
        NodeWithScore(node=TextNode(text="First doc"), score=0.5),
        NodeWithScore(node=TextNode(text="Second doc"), score=0.8),
    ]
    bundle = QueryBundle(query_str="test query")

    reranked = await reranker.apostprocess_nodes(nodes, bundle)

    assert len(reranked) == 2
    assert reranked[0].node.get_content() == "First doc"
    assert reranked[0].score > reranked[1].score


@pytest.mark.asyncio
async def test_aether_lora_reranker_peft_loading(mocker):
    mock_tokenizer = MagicMock()
    mocker.patch(
        "src.retrieval.reranker.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )

    mock_base_model = MagicMock()
    mocker.patch(
        "src.retrieval.reranker.AutoModelForSequenceClassification.from_pretrained",
        return_value=mock_base_model,
    )

    mock_peft_model = MagicMock()
    mock_fused_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_fused_model
    mocker.patch("peft.PeftModel.from_pretrained", return_value=mock_peft_model)

    reranker = AetherLoRAReranker(
        adapter_name_or_path="fake/adapter-path", fuse_adapter=True
    )

    assert reranker._model == mock_fused_model
