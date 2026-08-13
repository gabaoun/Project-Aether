import argparse
import os
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Upload trained LoRA adapter to Hugging Face Hub")
    parser.add_argument("--adapter_dir", type=str, default="./lora-bge-reranker-v2-m3")
    parser.add_argument("--repo_id", type=str, required=True, help="e.g. username/aether-bge-reranker-v2-m3-lora")
    parser.add_argument("--base_model_name", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--model_card_path", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face API token is required via --token, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN.")

    login(token=hf_token)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    tokenizer.push_to_hub(repo_id=args.repo_id, token=hf_token)

    base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model_name, num_labels=1)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    peft_model.push_to_hub(repo_id=args.repo_id, token=hf_token)

    if args.model_card_path and os.path.exists(args.model_card_path):
        api.upload_file(
            path_or_fileobj=args.model_card_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            token=hf_token,
        )


if __name__ == "__main__":
    main()
