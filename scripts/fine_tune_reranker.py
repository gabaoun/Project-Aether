import argparse
import os

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class RerankerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BAAI/bge-reranker-v2-m3 using PEFT/LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./lora-bge-reranker-v2-m3")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def load_triplet_or_pair_dataset(dataset_path: str | None) -> Dataset:
    if dataset_path and os.path.exists(dataset_path):
        return load_dataset("json", data_files=dataset_path, split="train")

    dummy_data = {
        "query": [
            "What is Project Aether?",
            "What is Project Aether?",
            "How does graph RAG work?",
            "How does graph RAG work?",
        ],
        "passage": [
            "Project Aether is an asynchronous event-driven RAG search engine in Python.",
            "Weather in Seattle is cold in winter.",
            "Graph RAG uses Knowledge Graphs to traverse document entity relationships.",
            "Cooking recipes require flour and sugar.",
        ],
        "label": [1.0, 0.0, 1.0, 0.0],
    }
    return Dataset.from_dict(dummy_data)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_dataset = load_triplet_or_pair_dataset(args.dataset_path)

    def preprocess_function(examples):
        queries = examples["query"]
        passages = examples["passage"]
        features = tokenizer(
            queries,
            passages,
            max_length=args.max_length,
            truncation=True,
            padding=False,
        )
        features["labels"] = examples["label"]
        return features

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
