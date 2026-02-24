# UGREEN License
#
# Copyright (c) 2025 UGREEN. All Rights Reserved.
#
# This software and associated documentation files (the "Software") are
# the proprietary information of UGREEN.
#
# The Software is provided solely for internal use within UGREEN
# and may not be copied, modified, distributed, or disclosed to any
# third party without prior written consent from UGREEN.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from pathlib import Path

from datasets import Dataset, load_dataset

from openllm_func_call_synthesizer.logger import logger


def convert_to_dataset(data: list[dict]) -> Dataset:
    """Convert a list of dictionaries to a Hugging Face Dataset.

    Args:
        data: A list of dictionaries.

    Returns:
        A Hugging Face Dataset.
    """
    load_dataset("json", data_files={"train": data})

    return Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})


def generate_fingerprint(dataset: "Dataset") -> str:
    import hashlib

    from datasets.fingerprint import Hasher
    def md5sum(path):
        m = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                m.update(chunk)
        return m.hexdigest()
    state = dataset.__dict__
    hasher = Hasher()
    for key in sorted(state):
        if key == "_fingerprint":
            continue
        hasher.update(key)
        hasher.update(state[key])
    # hash data files last modification timestamps as well
    for cache_file in sorted(dataset.cache_files):
        hasher.update(md5sum(cache_file))
    return hasher.hexdigest()


def persist_dataset_if_changed(dataset: Dataset, output_dir: Path, filename: str = "train") -> None:
    """Persist a Hugging Face Dataset to disk.

    Args:
        dataset: A Hugging Face Dataset.
        path: The path to save the dataset to.
    """
    fingerprint = dataset._fingerprint or generate_fingerprint(dataset)
    fingerprint_file = output_dir / f".{filename}.fingerprint"

    if fingerprint_file.exists() and fingerprint_file.read_text().strip() == fingerprint:
        logger.info("Dataset at %s unchanged; skipping serialization.", output_dir)
        return

    dataset.to_json(str(output_dir / f"{filename}.jsonl"), orient="records", lines=True)
    dataset.to_csv(str(output_dir / f"{filename}.csv"))
    dataset.to_parquet(str(output_dir / f"{filename}.parquet"))
    fingerprint_file.write_text(fingerprint)
    logger.info("Dataset saved to %s in jsonl/csv/parquet formats.", output_dir)


def format_openai(example: dict, system_prompt: str) -> dict:
    """Format an example for OpenAI.

    Args:
        example: A dictionary containing the example.

    Returns:
        A dictionary containing the formatted example.
    """
    import json

    message = json.loads(example["answer"])
    return {
        "messages": [
            {"role": "system", "content": system_prompt, "tool_calls": []},
            {"role": "user", "content": example["query"], "tool_calls": []},
            {
                "role": message["role"],
                "content": message["content"] or "",
                "tool_calls": message.get("tool_calls", []),
            },
        ],
        "tools": json.dumps(example["functions"], ensure_ascii=False)
        if not isinstance(example["functions"], str)
        else example["functions"],
    }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    file = Path(__file__).parent / "train.jsonl"
    dataset = load_dataset("json", data_files=file.as_posix())
    openai_format_dataset = dataset.map(
        format_openai,
        fn_kwargs={"system_prompt": "You are a helpful assistant."},
    ).remove_columns(dataset["train"].column_names)
    openai_format_dataset["train"].to_json("openai_format_dataset.jsonl", orient="records", lines=True)
    print(openai_format_dataset)
    """
    in LLaMA_FACTORY, DatasetInfo should be like this:

    "openai_format_dataset": {
        "file_name": "openai_format_dataset.jsonl",
        "formatting": "openai",
        "columns": {
        "messages": "messages",
        "tools": "tools"
        },
        "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
        "function_tag": "tool_calls"
        }
    }
    """
