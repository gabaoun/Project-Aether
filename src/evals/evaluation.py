import os
import json
import asyncio
from typing import List, Dict
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
)
from datasets import Dataset
from llama_index.core import VectorStoreIndex
from src.retrieval.retrieval_workflow import RetrievalWorkflow
from src.utils.config import settings

class RAGEvaluation:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.workflow = RetrievalWorkflow(index=index)

    async def generate_synthetic_dataset(self, test_questions: List[str]) -> Dataset:
        """Runs the workflow for a list of questions to prepare a RAGAS dataset."""
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [] # In a real scenario, this should be provided
        }
        
        for q in test_questions:
            result = await self.workflow.run(query=q)
            data["question"].append(q)
            data["answer"].append(result["answer"])
            data["contexts"].append([node.get_content() for node in result["source_nodes"]])
            data["ground_truth"].append("") # Placeholder
            
        return Dataset.from_dict(data)

    async def run_evaluation(self, test_questions: List[str]):
        """Evaluates the RAG engine using RAGAS metrics."""
        dataset = await self.generate_synthetic_dataset(test_questions)
        
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevance,
                context_precision,
            ],
        )
        
        report = result.to_pandas().to_json(orient="records")
        parsed_report = json.loads(report)
        
        # Save report
        with open("eval_report.json", "w") as f:
            json.dump(parsed_report, f, indent=4)
            
        print("Evaluation complete. Report saved to eval_report.json")
        return parsed_report

if __name__ == "__main__":
    # Example usage (requires an index)
    pass
