# MIT License
#
# Copyright (c) 2025, Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import asyncio
import json
import sys
from pathlib import Path

import hydra
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from fastmcp import Client
from omegaconf import DictConfig, OmegaConf
from rich import pretty

from openllm_func_call_synthesizer.core.critic import Critic
from openllm_func_call_synthesizer.core.synthesizer import (
    FunctionCallGenerator,
    QueryGenerator,
)
from openllm_func_call_synthesizer.logger import logger
from openllm_func_call_synthesizer.utils import (
    convert_to_openai_tools,
    tool_format_convert,
)
from openllm_func_call_synthesizer.utils.dataset_utils import format_openai


def _patch_argparse_help_for_py314() -> None:
    if sys.version_info < (3, 14):
        return

    original_check_help = argparse.ArgumentParser._check_help

    def _check_help(self, action):
        try:
            original_check_help(self, action)
        except ValueError as exc:
            if "badly formed help string" in str(exc) and not isinstance(action.help, str):
                action.help = str(action.help)
                original_check_help(self, action)
            else:
                raise

    argparse.ArgumentParser._check_help = _check_help


_patch_argparse_help_for_py314()


def _patch_datasets_dill_for_py314() -> None:
    """Allow datasets' dill shim to handle the Py3.14 dict pickler API."""

    if sys.version_info < (3, 14):
        return

    try:
        from datasets.utils import _dill as datasets_dill  # type: ignore
    except Exception:
        return

    if getattr(datasets_dill.Pickler, "_py314_batch_patch", False):
        return

    batch_setitems = datasets_dill.Pickler.__dict__.get("_batch_setitems")
    code = getattr(batch_setitems, "__code__", None)
    if batch_setitems is None or code is None or code.co_argcount >= 3:
        return

    import dill  # local import to avoid global dependency if datasets unused

    def _patched_batch_setitems(self, items, obj=None):
        if getattr(self, "_legacy_no_dict_keys_sorting", False):
            return dill.Pickler._batch_setitems(self, items, obj)

        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))

        return dill.Pickler._batch_setitems(self, items, obj)

    datasets_dill.Pickler._batch_setitems = _patched_batch_setitems
    datasets_dill.Pickler._py314_batch_patch = True


load_dotenv(override=True)


def _serialize_nested_for_parquet(ds):
    """Convert nested (list/dict) columns to JSON strings so pyarrow can write parquet.

    This inspects a small sample of the dataset to find nested columns and maps
    a serializer over the full dataset only when needed to avoid unnecessary work.
    """
    try:
        # import here to avoid hard dependency at module import time
        from datasets import Dataset
    except Exception:
        Dataset = None

    # Only operate on HuggingFace Dataset objects
    if Dataset is None or not isinstance(ds, Dataset):
        return ds

    length = len(ds)
    sample_n = min(10, length)
    if sample_n <= 0:
        return ds

    nested_cols = set()
    for ex in ds.select(range(sample_n)):
        for k, v in ex.items():
            if isinstance(v, list | dict):
                nested_cols.add(k)

    if not nested_cols:
        return ds

    def _serializer(example):
        for k in nested_cols:
            v = example.get(k)
            if isinstance(v, list | dict):
                example[k] = json.dumps(v, ensure_ascii=False)
        return example

    # Map to serialize nested fields. Keep it single-threaded to avoid pickling issues.
    return ds.map(_serializer)


async def get_mcp_tools(mcp_cfg: dict) -> list[dict]:
    """Get tools from MCP server."""
    try:
        client = Client(**mcp_cfg)
        async with client:
            tools = await client.list_tools()
    except Exception as exc:
        logger.error("Failed to get tool list from MCP server %s", mcp_cfg, exc_info=exc)
        raise
    return tools


def choose_tools(openai_format_tools: dict, target_names: list[str]):
    if not target_names:
        target_names = [
            "search_photos",
            "create_album",
            "get_album_list",
            "music_play_control",
            "music_search_control",
            "music_settings_control",
            "video_search_control",
            "video_play_control",
            "get_system_info",
        ]

    # Iterate through openai_format_tools['tools'] and keep only function names found in target_names
    filtered_tools = []
    for tool in openai_format_tools["tools"]:
        # Ensure the tool has a function payload with a name field
        function_info = tool.get("function", {})
        func_name = function_info.get("name", None)
        if func_name in target_names:
            filtered_tools.append(tool)

    # Replace openai_format_tools with a filtered version that preserves the format
    openai_format_tools2 = {"tools": filtered_tools}

    logger.info(
        "------------openai_format_tools choosed------------ %s %s", type(openai_format_tools2), openai_format_tools2
    )

    return openai_format_tools2


def generate_query_dataset(cfg: DictConfig, function_docs: list[dict]):
    """Generate a dataset of queries for function calls.

    Args:
        cfg (DictConfig): Configuration object containing settings for query generation.
        function_docs (list[dict]): List of function documentation dictionaries.

    Raises:
        FileNotFoundError: If the specified function documentation file is not found.
    """
    # choose part tools to generate query
    if cfg.synthesizer.choose_part_tools:
        logger.info("--------cfg.synthesizer.choose_part_tools----- %s", cfg.synthesizer.choose_part_tools)
        if cfg.synthesizer.choose_part_tools:
            function_docs_choosed = choose_tools(function_docs, cfg.synthesizer.choose_part_tools)
            logger.info("------------function_docs_choosed------------ %s", function_docs_choosed)
            function_docs = function_docs_choosed
    data_file = cfg.synthesizer.query_generation.function_docs

    query_generator_cfg = cfg.synthesizer.query_generation
    OmegaConf.set_struct(query_generator_cfg, False)
    if Path(data_file).exists():
        # Load the tool definitions (with optional queries) from function_docs.json
        with open(data_file) as f:
            json_tools_data = json.load(f)

        # Build a mapping of tool name to its query seed(s)
        tool_queries_map = {}
        for tool in json_tools_data.get("tools", []):
            tool_name = tool.get("name")
            if tool_name and "query" in tool:
                tool_queries_map[tool_name] = tool["query"]

        logger.info("---------tool_queries_map------------ %s", tool_queries_map)

    dataset_records = []
    for tool in function_docs["tools"]:
        # Get the tool name for the current entry
        function_info = tool.get("function", {})
        tool_name = function_info.get("name")

        # Lookup the query list/value mapped to this tool name
        if "tool_queries_map" in locals():
            seed_queries = tool_queries_map.get(tool_name, None)
        else:
            seed_queries = None

        logger.info("---------tool: %s, seed_queries: %s------------", tool_name, seed_queries)

        tool_copy = dict(tool)
        function_repr = json.dumps(tool_copy, ensure_ascii=False, indent=2)

        if isinstance(seed_queries, list):
            logger.info("---------is list------------ %s", seed_queries)
            for seed in seed_queries:
                logger.info("---------seed------------ %s", seed)
                dataset_records.append({"function": function_repr, "query": seed})
        elif isinstance(seed_queries, str) and seed_queries.strip():
            dataset_records.append({"function": function_repr, "query": seed_queries})
        else:
            dataset_records.append({"function": function_repr})
    data = dataset_records
    logger.info("#################### data #################### %s", data)
    pretty.pprint(data)
    # Loop over configured languages to generate multilingual query variations
    languages = query_generator_cfg.get("languages", ["English"])
    output_datasets = []

    for language in languages:
        for name, provider in query_generator_cfg.providers.items():
            logger.info("provider: %s, language: %s", name, language)
            for model in provider.models:
                logger.info("model: %s", model)
                # Instantiate generator with language
                qg = QueryGenerator(
                    model_name=model,
                    language=language,
                    backend=provider.backend,
                    backend_params=provider.backend_params,
                )
                # Generate records by iterating through examples and their variations
                queries = qg(dataset=data)
                ds = queries.dataset.map(lambda x, name=name, model=model: {"provider": name, "model": model})

                # No need to flatten explicitly; iterate over dataset
                # Collect this provider/model/language dataset
                output_datasets.append(ds)
    # Combine and save all provider/model datasets
    combined = concatenate_datasets(output_datasets)

    # Ensure output directory exists
    output_dir = Path(query_generator_cfg.get("output_dir", "data")) / cfg.synthesizer.query_generation.name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save in multiple formats
    # Save JSON Lines under 'train.jsonl' so HuggingFace load_dataset can load it as the 'train' split
    combined.to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    combined.to_csv(str(output_dir / "train.csv"))
    combined.to_parquet(str(output_dir / "train.parquet"))
    logger.info("Dataset saved to %s in jsonl, xlsx, parquet formats.", output_dir)


def generate_function_call_dataset(cfg: DictConfig, mcp_tools: list[dict]):
    # Load the function dataset
    function_call_cfg = cfg.synthesizer.function_call_generation
    function_dataset_path = Path(function_call_cfg.function_dataset)
    if not function_dataset_path.exists():
        raise FileNotFoundError(f"File {function_dataset_path} not found")

    if function_dataset_path.is_file():
        data_files = {"train": str(function_dataset_path)}
    else:
        data_files = {"train": str(function_dataset_path / "train.jsonl")}

    dataset = load_dataset("json", data_files=data_files)

    fc_kwargs = OmegaConf.to_container(function_call_cfg.provider, resolve=True)
    function_docs = tool_format_convert(mcp_tools, fc_kwargs["model_name"])

    generation_params = fc_kwargs.get("generation_params", {})
    generation_params.update({"tools": function_docs["tools"]})
    fc_kwargs["generation_params"] = generation_params

    function_call_generator = FunctionCallGenerator(**fc_kwargs)
    max_num = function_call_cfg.max_num
    if max_num > 0:
        dataset = dataset["train"].select(range(max_num))
    else:
        dataset = dataset["train"]
    dataset = dataset.map(lambda x: {"functions": json.dumps(function_docs["tools"], ensure_ascii=False)})

    fcg = function_call_generator(dataset=dataset)

    output_dir = Path(function_call_cfg.output_dir) / function_call_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    fcg.dataset.to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    fcg.dataset.to_csv(str(output_dir / "train.csv"))
    fcg.dataset.to_parquet(str(output_dir / "train.parquet"))
    logger.info("Dataset saved to %s in train.jsonl, csv, parquet formats.", output_dir)


def critic_function_call_dataset(cfg: DictConfig):
    critic_cfg = cfg.synthesizer.critic
    function_call_dataset_path = Path(critic_cfg.function_call_dataset)
    if not function_call_dataset_path.exists():
        raise FileNotFoundError(f"File {function_call_dataset_path} not found")

    if function_call_dataset_path.is_file():
        data_files = {"train": str(function_call_dataset_path)}
    else:
        data_files = {"train": str(function_call_dataset_path / "train.jsonl")}

    dataset = load_dataset("json", data_files=data_files)
    cg_args = OmegaConf.to_container(cfg.synthesizer.critic.provider, resolve=True)
    cg_args["query_field"] = critic_cfg.query_field
    cg_args["task_prompt_field"] = critic_cfg.task_prompt_field
    cg_args["label_field"] = critic_cfg.label_field
    cg_args["functions_field"] = critic_cfg.functions_field
    cg_args["response_field"] = critic_cfg.response_field
    cg_args["use_gt"] = critic_cfg.use_ground_truth if "use_ground_truth" in critic_cfg else False
    critic_generate = Critic(**cg_args)
    max_num = cfg.synthesizer.function_call_generation.max_num
    if max_num > 0:
        dataset = dataset["train"].select(range(max_num))
    else:
        dataset = dataset["train"]

    output_dir = Path(critic_cfg.output_dir) / critic_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cg = critic_generate(dataset=dataset)

    cg.dataset.to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    cg.dataset.to_csv(str(output_dir / "train.csv"))
    cg.dataset.to_parquet(str(output_dir / "train.parquet"))

    logger.info("Dataset saved to %s in train.jsonl, csv, parquet formats.", output_dir)


def create_llama_factory_compatible_dataset(cfg: DictConfig):
    llama_factory_cfg = cfg.synthesizer.llama_factory
    critic_dataset_path = Path(llama_factory_cfg.critic_dataset)
    if not critic_dataset_path.exists():
        raise FileNotFoundError(f"File {critic_dataset_path} not found")
    if critic_dataset_path.is_file():
        data_files = {"train": str(critic_dataset_path)}
    else:
        data_files = {"train": str(critic_dataset_path / "train.jsonl")}
    logger.info("%s", data_files)
    dataset = load_dataset("json", data_files=data_files)
    if llama_factory_cfg.score_field in dataset["train"].column_names:
        dataset = dataset.filter(lambda x: x[llama_factory_cfg.score_field] >= llama_factory_cfg.score_threshold)

    openai_format_dataset = dataset.map(
        format_openai,
        fn_kwargs={"system_prompt": llama_factory_cfg.system_prompt},
    ).remove_columns(dataset["train"].column_names)
    output_dir = Path(llama_factory_cfg.output_dir) / llama_factory_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if llama_factory_cfg.split_ratio > 0 and llama_factory_cfg.split_ratio < 1:
        openai_format_dataset = openai_format_dataset["train"].train_test_split(
            test_size=1 - llama_factory_cfg.split_ratio
        )

    openai_format_dataset["train"].to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    openai_format_dataset["train"].to_csv(str(output_dir / "train.csv"))
    openai_format_dataset["train"].to_parquet(str(output_dir / "train.parquet"))
    if "test" in openai_format_dataset:
        openai_format_dataset["test"].to_json(str(output_dir / "test.jsonl"), orient="records", lines=True)
        openai_format_dataset["test"].to_csv(str(output_dir / "test.csv"))
        openai_format_dataset["test"].to_parquet(str(output_dir / "test.parquet"))


def create_verl_compatible_dataset(cfg: DictConfig):
    verl_cfg = cfg.synthesizer.verl
    critic_dataset_path = Path(verl_cfg.critic_dataset)
    if not critic_dataset_path.exists():
        raise FileNotFoundError(f"File {critic_dataset_path} not found")
    dataset = load_dataset("json", data_files={"train": str(critic_dataset_path / "train.jsonl")})
    if verl_cfg.score_field in dataset["train"].column_names:
        dataset = dataset.filter(lambda x: x[verl_cfg.score_field] < verl_cfg.score_threshold)

    openai_format_dataset = dataset.map(
        format_openai,
        fn_kwargs={"system_prompt": verl_cfg.system_prompt},
    ).remove_columns(dataset["train"].column_names)
    output_dir = Path(verl_cfg.output_dir) / verl_cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if verl_cfg.split_ratio > 0 and verl_cfg.split_ratio < 1:
        openai_format_dataset = openai_format_dataset["train"].train_test_split(test_size=1 - verl_cfg.split_ratio)

    openai_format_dataset["train"].to_json(str(output_dir / "train.jsonl"), orient="records", lines=True)
    openai_format_dataset["train"].to_csv(str(output_dir / "train.csv"))
    openai_format_dataset["train"].to_parquet(str(output_dir / "train.parquet"))
    if "test" in openai_format_dataset:
        openai_format_dataset["test"].to_json(str(output_dir / "test.jsonl"), orient="records", lines=True)
        openai_format_dataset["test"].to_csv(str(output_dir / "test.csv"))
        openai_format_dataset["test"].to_parquet(str(output_dir / "test.parquet"))


@hydra.main(config_path="../examples/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pretty.pprint("loading config:")
    pretty.pprint(cfg)
    logger.info("loading tools from MCP server:")
    # Use asyncio.run to ensure a running event loop (avoids get_event_loop() RuntimeError on Py3.14)
    mcp_tools = asyncio.run(get_mcp_tools(mcp_cfg=cfg.synthesizer.mcp_servers["ugreen_mcp"]))
    logger.info("------------mcp_tools------------ %s", mcp_tools)
    openai_format_tools = convert_to_openai_tools(mcp_tools)
    pretty.pprint(openai_format_tools)
    synth_cfg = cfg.synthesizer
    logger.info("synth_config: ")
    pretty.pprint(synth_cfg)

    if cfg.synthesizer.query_generation.enable:
        generate_query_dataset(cfg, function_docs=openai_format_tools)
    if cfg.synthesizer.function_call_generation.enable:
        generate_function_call_dataset(cfg, mcp_tools=mcp_tools)
    if cfg.synthesizer.critic.enable:
        critic_function_call_dataset(cfg)
    if cfg.synthesizer.llama_factory.enable:
        create_llama_factory_compatible_dataset(cfg)
    if cfg.synthesizer.verl.enable:
        create_verl_compatible_dataset(cfg)


if __name__ == "__main__":
    main()
