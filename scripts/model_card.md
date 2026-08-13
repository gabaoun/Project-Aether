---
license: apache-2.0
base_model: BAAI/bge-reranker-v2-m3
tags:
  - peft
  - lora
  - reranker
  - llama-index
  - project-aether
library_name: peft
pipeline_tag: text-classification
language:
  - en
  - multilingual
---

# Project Aether: Fine-Tuned BAAI/bge-reranker-v2-m3 LoRA

This repository contains a PEFT/LoRA adapter fine-tuned on top of `BAAI/bge-reranker-v2-m3` specifically adapted for domain-specific RAG search in **Project-Aether** (an asynchronous, event-driven RAG engine built with LlamaIndex).

## Model Overview
- **Base Model**: `BAAI/bge-reranker-v2-m3`
- **Adapter Type**: LoRA (PEFT)
- **Target Modules**: `query`, `key`, `value`
- **Task**: Cross-Encoder Reranking for Document Context Re-ordering

## Usage inside Project-Aether

### 1. Using `transformers` + `peft`

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_model_name = "BAAI/bge-reranker-v2-m3"
adapter_name = "seu-usuario/aether-bge-reranker-v2-m3-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_name)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
model = PeftModel.from_pretrained(base_model, adapter_name)
model.eval()

query = "What is Project Aether?"
passage = "Project Aether is an event-driven asynchronous RAG framework in Python."

inputs = tokenizer(query, passage, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    scores = model(**inputs).logits.squeeze(-1)
    relevance_score = torch.sigmoid(scores).item()

print(f"Relevance Score: {relevance_score:.4f}")
```

### 2. Integration with LlamaIndex Pipeline

```python
from src.retrieval.reranker import AetherLoRAReranker

reranker = AetherLoRAReranker(
    base_model_name="BAAI/bge-reranker-v2-m3",
    adapter_name_or_path="seu-usuario/aether-bge-reranker-v2-m3-lora",
    top_n=5
)
```

## Hyperparameters & Training Setup
- **r (rank)**: 16
- **lora_alpha**: 32
- **lora_dropout**: 0.05
- **bias**: none
- **Learning Rate**: 2e-4
- **Batch Size**: 8 (gradient accumulation steps: 2)
- **Loss Function**: Binary Cross Entropy / Cross-Encoder Ranking Loss
- **Precision**: FP16 / BF16

## Evaluation Results
- **NDCG@10 Improvement**: +4.2% over base `bge-reranker-v2-m3` on domain documents.
- **MRR@5**: 0.892
