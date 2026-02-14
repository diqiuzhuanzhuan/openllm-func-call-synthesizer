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


from datasets import Dataset, load_dataset


def convert_to_dataset(data: list[dict]) -> Dataset:
    """Convert a list of dictionaries to a Hugging Face Dataset.

    Args:
        data: A list of dictionaries.

    Returns:
        A Hugging Face Dataset.
    """
    load_dataset("json", data_files={"train": data})

    return Dataset.from_dict({k: [dic[k] for dic in data] for k in data[0]})


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
