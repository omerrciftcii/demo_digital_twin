"""Explicitly download PUBLIC embedding weights; never load patient documents.

Run after installing the semantic extra. Startup only reads this local cache.
"""

import argparse

# Only the PyTorch SentenceTransformer path is used on the CPU. Do not fetch
# duplicate ONNX/OpenVINO/TF exports or alternative quantizations for this model.
MODEL_FILES = [
    "config.json", "modules.json", "config_sentence_transformers.json",
    "sentence_bert_config.json", "1_Pooling/config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model",
    "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()
    from huggingface_hub import list_repo_files, snapshot_download

    names = list_repo_files(repo_id=args.model)
    safe_weights = [name for name in names if "/" not in name and (
        name == "model.safetensors" or name == "model.safetensors.index.json"
        or name.startswith("model-") and name.endswith(".safetensors")
    )]
    torch_weights = [name for name in names if "/" not in name and (
        name == "pytorch_model.bin" or name == "pytorch_model.bin.index.json"
        or name.startswith("pytorch_model-") and name.endswith(".bin")
    )]
    weights = safe_weights or torch_weights
    if not weights:
        raise SystemExit("No supported root-level PyTorch model weights found")
    location = snapshot_download(repo_id=args.model, allow_patterns=[*MODEL_FILES, *weights])
    print(f"Public embedding model cached at {location}")
    print("Restart the application to enable local semantic indexing.")


if __name__ == "__main__":
    main()
