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

import json

from bespokelabs import curator
from bespokelabs.curator.log import logger
from pydantic import BaseModel, Field
from rich import pretty
from xxhash import xxh64

from openllm_func_call_synthesizer.core.formatter import QUERY_GENERATE_SYSTEM_HEADER
from openllm_func_call_synthesizer.utils import extract_format, parse_hermes_tool_calls


class FunctionCallGenerator(curator.LLM):
    """A simple function calling generator."""

    return_completions_object = True
    debug = False

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions
        # return f"""
        # {input["query"]}
        # """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input["query"].strip()},
        ]
        return messages

    def _parse_function_call(self, raw_output: dict) -> dict:
        parsed = []
        for call in raw_output:
            # Handle standard format with "function" wrapper
            if "function" in call:
                func = call["function"]
                name = func.get("name")
                args_str = func.get("arguments", "{}")
            else:
                # Handle flat format
                name = call.get("name")
                args_str = call.get("arguments", "{}")

            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args = args_str  # fallback
            parsed.append({"name": name, "arguments": args})

        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _deduplicate_input_ls(self, input_ls):
        """
        Deduplicate input_ls based on the fields: prompt, function_call, answer.
        If all three fields are identical (ignoring 'tool_call' id), keep only one instance.

        Due to the fact that the 'answer' field contains the 'id' of 'tool_calls',
        and each call has a different id, deduplication fails.
        Therefore, when comparing, the 'id' field under 'tool_call' must be ignored!

        """

        def norm_answer(answer):
            # the format of answer is a string of json, so we need to parse it first
            try:
                data = json.loads(answer)
            except Exception:
                return answer  # just return the original answer if it's not a valid json
            # remove the id field under tool_calls
            if isinstance(data, dict) and "tool_calls" in data and isinstance(data["tool_calls"], list):
                for tc in data["tool_calls"]:
                    if isinstance(tc, dict) and "id" in tc:
                        tc.pop("id")
            try:
                return json.dumps(data, ensure_ascii=False, sort_keys=True)
            except Exception:
                return answer

        seen = set()
        deduped = []
        for item in input_ls:
            key = (item.get("prompt", ""), item.get("function_call", ""), norm_answer(item.get("answer", "")))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def parse(self, input: dict, response) -> list:
        """Parse each choice in the response to extract the function call or the message."""
        input_ls = []
        prompt = self.prompt(input)
        print("--------------choices response------------------", response["choices"])
        for choice in response["choices"]:
            this_input = dict(input)  # make a shallow copy
            this_input["prompt"] = prompt

            message = choice.get("message", {})
            # Convert message to dict if it's an object (like litellm Message object)
            if hasattr(message, "model_dump"):
                message = message.model_dump()
            elif hasattr(message, "__dict__"):
                message = message.__dict__

            this_input["raw_output"] = message
            parsed_fc = parse_hermes_tool_calls(message)
            # now, we always serialize the function call to a json string
            this_input["function_call"] = json.dumps(parsed_fc, ensure_ascii=False) if parsed_fc else ""
            this_input["answer"] = json.dumps(message, ensure_ascii=False, indent=2)
            if self.debug:
                if "answer" in this_input:
                    pretty.pprint("answer: ")
                    pretty.pprint(this_input["answer"])
                if "function_call" in this_input:
                    pretty.pprint("function_call: ")
                    pretty.pprint(this_input["function_call"])
                pretty.pprint("ground_truth: ")
                pretty.pprint(this_input.get("ground_truth", ""))
            input_ls.append(this_input)
        # Deduplicate before return
        if len(input_ls) > 1:
            if self.debug:
                print(" ------------ input list ------------ ", input_ls)
            input_ls = self._deduplicate_input_ls(input_ls)
            if self.debug:
                print(" ------------ deduped input list ------------ ", input_ls)
        return input_ls


class QueryFunc(BaseModel):
    query: str = Field(..., description="The natural language query")
    function: str = Field(..., description="The function name to call")
    dimension: str = Field(..., description="The variation dimension")
    language: str = Field(..., description="The query language")


class QueryFuncItem(BaseModel):
    item: QueryFunc = Field(..., description="The query function item")


class QueryGenerator(curator.LLM):
    """A simple query generator."""

    return_completions_object = True

    def __init__(self, model_name: str = None, backend: str = None, language: str = "English", **kwargs):
        """Initialize with optional language for generation."""
        super().__init__(model_name=model_name, backend=backend, **kwargs)
        self.language = language

    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False):
        from xxhash import xxh64

        fingerprint = super()._hash_fingerprint(dataset_hash, disable_cache)
        fingerprint = f"{fingerprint}_{xxh64(self.language.encode('utf-8')).hexdigest()}"
        logger.info(f"Curator Cache Fingerprint: {fingerprint}")
        return fingerprint

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the query."""
        seed_query = input.get("query", "")
        return QUERY_GENERATE_SYSTEM_HEADER.format(
            language=self.language, function=input["function"], seed_query=seed_query, function_name=input["function"]
        )

    def parse(self, input: dict, response) -> list[dict]:
        """Parse the response to extract the query."""

        query = extract_format(format="json", content=response["choices"][0]["message"]["content"])
        function_hash = xxh64(str(input["function"]).encode("utf-8")).hexdigest()
        # Build a list of query variation records with metadata
        output = [
            {
                "query": ele["query"],
                "dimension": ele["dimension"],
                "language": self.language,
                "function": input["function"],
                "function_hash": function_hash,
            }
            for ele in query["variations"]
        ]
        return output


class ConversionGenerator(curator.LLM):
    """A simple conversion generator."""

    return_completions_object = True

    def prompt(self, input: dict) -> str:
        """The prompt is used to generate the conversion."""
        return f"""You are a conversion generation expert. Given the user request:
        {input["user_request"]}.
        Generate a conversion that can be used to satisfy the user request.
        """

    def parse(self, input: dict, response) -> dict:
        """Parse the response to extract the conversion."""
        input["conversion"] = response["choices"][0]["message"]["content"]
        return input


if __name__ == "__main__":
    qg = QueryGenerator()
