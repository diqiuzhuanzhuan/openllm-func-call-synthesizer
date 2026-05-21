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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

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

    def prompt(self, input: dict[str, Any] | BaseModel) -> dict[str, Any] | BaseModel:
        """The prompt is used to generate the function call."""
        if isinstance(input, BaseModel):
            input = input.model_dump()
        # Prepare a readable listing of available functions
        # return f"""
        # {input["query"]}
        # """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input["query"].strip()},
        ]
        return cast(dict[str, Any] | BaseModel, messages)

    def _parse_function_call(self, raw_output: list[dict[str, Any]]) -> str:
        parsed: list[dict[str, Any]] = []
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

    def _deduplicate_input_ls(self, input_ls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Deduplicate input_ls based on the fields: prompt, function_call, answer.
        If all three fields are identical (ignoring 'tool_call' id), keep only one instance.

        Due to the fact that the 'answer' field contains the 'id' of 'tool_calls',
        and each call has a different id, deduplication fails.
        Therefore, when comparing, the 'id' field under 'tool_call' must be ignored!

        """

        def norm_answer(answer: Any) -> Any:
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

        def _to_hashable(value: Any) -> Any:
            if isinstance(value, list | dict):
                try:
                    return json.dumps(value, ensure_ascii=False, sort_keys=True)
                except TypeError:
                    return str(value)
            return value

        for item in input_ls:
            key = (
                _to_hashable(item.get("prompt", "")),
                _to_hashable(item.get("function_call", "")),
                norm_answer(item.get("answer", "")),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    def parse(
        self, input: dict[str, Any] | BaseModel, response: dict[str, Any] | BaseModel
    ) -> dict[str, Any] | BaseModel:
        """Parse each choice in the response to extract the function call or the message."""
        input_dict = input.model_dump() if isinstance(input, BaseModel) else dict(input)
        response_dict = response.model_dump() if isinstance(response, BaseModel) else response
        input_ls = []
        prompt = self.prompt(input_dict)
        print("--------------choices response------------------", response_dict["choices"])
        for choice in response_dict["choices"]:
            this_input = dict(input_dict)  # make a shallow copy
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
        return cast(dict[str, Any] | BaseModel, input_ls)


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

    def __init__(
        self, model_name: str = "", backend: str | None = None, language: str = "English", **kwargs: Any
    ) -> None:
        """Initialize with optional language for generation."""
        super().__init__(model_name=model_name, backend=backend, **kwargs)
        self.language = language

    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False) -> str:
        fingerprint = super()._hash_fingerprint(dataset_hash, disable_cache)
        fingerprint = f"{fingerprint}_{xxh64(self.language.encode('utf-8')).hexdigest()}"
        logger.info(f"Curator Cache Fingerprint: {fingerprint}")
        return fingerprint

    def prompt(self, input: dict[str, Any] | BaseModel) -> dict[str, Any] | BaseModel:
        """The prompt is used to generate the query."""
        if isinstance(input, BaseModel):
            input = input.model_dump()
        seed_query = input.get("query", "")
        return cast(
            dict[str, Any] | BaseModel,
            QUERY_GENERATE_SYSTEM_HEADER.format(
                language=self.language,
                function=input["function"],
                seed_query=seed_query,
                function_name=input["function"],
            ),
        )

    def parse(
        self, input: dict[str, Any] | BaseModel, response: dict[str, Any] | BaseModel
    ) -> dict[str, Any] | BaseModel:
        """Parse the response to extract the query."""
        input_dict = input.model_dump() if isinstance(input, BaseModel) else dict(input)
        response_dict = response.model_dump() if isinstance(response, BaseModel) else response

        query = extract_format(format="json", content=response_dict["choices"][0]["message"]["content"])
        function_hash = xxh64(str(input_dict["function"]).encode("utf-8")).hexdigest()
        # Build a list of query variation records with metadata
        output = [
            {
                "query": ele["query"],
                "dimension": ele["dimension"],
                "language": self.language,
                "function": input_dict["function"],
                "function_hash": function_hash,
            }
            for ele in query["variations"]
        ]
        return cast(dict[str, Any] | BaseModel, output)


DEFAULT_USER_SYSTEM_PROMPT = (
    "You are role-playing as the HUMAN participant in a conversation with an AI assistant. "
    "Base your goals on the scenario description you are given. Ask natural follow-up questions, "
    "refer to previous assistant replies, and keep your responses concise and realistic."
)

DEFAULT_ASSISTANT_SYSTEM_PROMPT = (
    "You are the helpful AI assistant in a multi-turn conversation. Respond helpfully and "
    "proactively address the human's requests from the scenario description."
)


@dataclass
class ConversationTurn:
    role: str
    content: str


class ConversationRoleLLM(curator.LLM):
    """Internal helper LLM subclass that generates the next message for a conversation role."""

    return_completions_object = True

    def __init__(
        self,
        *,
        model_name: str,
        target_role: str,
        system_prompt: str,
        backend: str | None = None,
        generation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, backend=backend, generation_params=generation_params, **kwargs)
        self.target_role = target_role
        self.system_prompt = system_prompt

    def prompt(self, input: dict[str, Any] | BaseModel) -> dict[str, Any] | BaseModel:
        input_dict = input.model_dump() if isinstance(input, BaseModel) else dict(input)
        scenario = str(input_dict["scenario"])
        history = input_dict.get("history", [])
        if not isinstance(history, list):
            raise ValueError("history must be a list of messages")
        messages: list[dict[str, str]] = [{"role": "system", "content": f"{self.system_prompt}\nScenario: {scenario}"}]
        messages.extend(history)
        if self.target_role == "user":
            instruction = "Write the human participant's next natural message."
        else:
            instruction = "Respond as the assistant, addressing the latest human turn."
        messages.append({"role": "user", "content": instruction})
        return cast(dict[str, Any] | BaseModel, messages)

    def parse(
        self, input: dict[str, Any] | BaseModel, response: dict[str, Any] | BaseModel
    ) -> dict[str, Any] | BaseModel:
        input_dict = input.model_dump() if isinstance(input, BaseModel) else dict(input)
        response_dict = response.model_dump() if isinstance(response, BaseModel) else response
        content = response_dict["choices"][0]["message"]["content"].strip()
        input_dict["next_turn"] = {"role": self.target_role, "content": content}
        return input_dict


class ConversationRunner(Protocol):
    def __call__(self, rows: list[dict[str, Any]]) -> Any: ...


class ConversationGenerator:
    """Simulate a two-LLM conversation using curator-powered role models."""

    def __init__(
        self,
        *,
        user_model_name: str,
        assistant_model_name: str,
        max_turns: int = 6,
        user_backend: str | None = None,
        assistant_backend: str | None = None,
        user_generation_params: dict[str, Any] | None = None,
        assistant_generation_params: dict[str, Any] | None = None,
        user_system_prompt: str | None = None,
        assistant_system_prompt: str | None = None,
        human_llm: ConversationRunner | None = None,
        assistant_llm: ConversationRunner | None = None,
    ) -> None:

        if max_turns < 2:
            raise ValueError("max_turns must be at least 2 so both roles can speak")
        self.max_turns = max_turns
        self.human_llm = cast(
            ConversationRunner,
            human_llm
            or ConversationRoleLLM(
                model_name=user_model_name,
                backend=user_backend,
                generation_params=user_generation_params,
                target_role="user",
                system_prompt=user_system_prompt or DEFAULT_USER_SYSTEM_PROMPT,
            ),
        )
        self.assistant_llm = cast(
            ConversationRunner,
            assistant_llm
            or ConversationRoleLLM(
                model_name=assistant_model_name,
                backend=assistant_backend,
                generation_params=assistant_generation_params,
                target_role="assistant",
                system_prompt=assistant_system_prompt or DEFAULT_ASSISTANT_SYSTEM_PROMPT,
            ),
        )

    def _run_role_model(self, llm: ConversationRunner, scenario: str, history: list[dict[str, str]]) -> dict[str, str]:
        row = {"scenario": scenario, "history": history}
        response = llm([row])
        output_rows = response.dataset.to_list()
        return output_rows[0]["next_turn"]

    @staticmethod
    def _validate_turn(turn: dict[str, str], expected_role: str) -> ConversationTurn:
        if not isinstance(turn, dict):
            raise ValueError(f"Conversation turn must be a dict, got {type(turn).__name__}")

        role = turn.get("role")
        content = turn.get("content")

        if not isinstance(role, str):
            raise ValueError(f"Expected role '{expected_role}', got '{role}'")
        if role != expected_role:
            raise ValueError(f"Expected role '{expected_role}', got '{role}'")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Conversation turn content for role '{expected_role}' must be a non-empty string")

        return ConversationTurn(role=role, content=content.strip())

    def generate(self, user_request: str, seed_history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        scenario_context = user_request.strip()
        history = [ConversationTurn(**turn) for turn in (seed_history or [])]

        while len(history) < self.max_turns:
            if not history or history[-1].role == "assistant":
                next_turn = self._run_role_model(self.human_llm, scenario_context, [t.__dict__ for t in history])
                history.append(self._validate_turn(next_turn, "user"))

            if len(history) >= self.max_turns:
                break

            next_turn = self._run_role_model(self.assistant_llm, scenario_context, [t.__dict__ for t in history])
            history.append(self._validate_turn(next_turn, "assistant"))

        return [turn.__dict__ for turn in history]

    def generate_dataset(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dataset_rows: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            scenario = (record.get("scenario") or record.get("user_request") or "").strip()
            if not scenario:
                raise ValueError(f"Conversation dataset record {index} is missing a non-empty scenario")

            seed_history = record.get("seed_history") or []
            if not isinstance(seed_history, list):
                raise ValueError(f"Conversation dataset record {index} has invalid seed_history: expected list")

            conversation = self.generate(scenario, seed_history=seed_history)
            dataset_rows.append(
                {
                    "scenario": scenario,
                    "seed_history": json.dumps(seed_history, ensure_ascii=False),
                    "conversation": json.dumps(conversation, ensure_ascii=False),
                    "turn_count": len(conversation),
                }
            )

        return dataset_rows


class ToolCallingLoop:
    """Run an agentic tool-calling loop, dispatching tool calls via the MCP protocol.

    The *llm_callable* receives the current messages list and must return a single
    assistant message dict (``{"role": "assistant", ...}``).  When that message
    contains a ``tool_calls`` list the loop executes each call in order via the
    provided ``fastmcp.Client``, appends a ``{"role": "tool", ...}`` message for
    each result, and calls the LLM again.  The loop stops when the assistant
    returns a message without ``tool_calls`` or after *max_iterations* LLM calls.

    Args:
        llm_callable: ``(messages: list[dict]) -> dict`` – returns the assistant reply.
        mcp_client: A ``fastmcp.Client`` instance used to call tools via MCP.
        max_iterations: Upper bound on the number of LLM calls (default 10).
    """

    def __init__(
        self,
        llm_callable: Callable[[list[dict]], dict],
        mcp_client: Any,
        max_iterations: int = 10,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.llm = llm_callable
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations

    @staticmethod
    def _extract_content(result: Any) -> str:
        """Extract a plain-text string from a ``fastmcp`` ``CallToolResult``."""
        parts = [block.text for block in result.content if hasattr(block, "text")]
        text = "\n".join(parts) if parts else (str(result.data) if result.data is not None else "")
        return f"Error: {text}" if result.is_error else text

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute one tool call via MCP and return its result as a string."""
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments_raw = function.get("arguments", "{}")

        try:
            arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
        except json.JSONDecodeError:
            arguments = {}

        try:
            result = await self.mcp_client.call_tool(name, arguments, raise_on_error=False)
            return self._extract_content(result)
        except Exception as exc:
            return f"Error calling tool '{name}': {exc}"

    async def run(self, messages: list[dict]) -> list[dict]:
        """Run the tool-calling loop and return the complete messages list.

        Opens the MCP client for the duration of the loop so the connection is
        shared across all tool calls within a single ``run()`` invocation.

        Args:
            messages: Seed messages (system / user / …).

        Returns:
            The full conversation including assistant replies and tool results.
        """
        messages = list(messages)

        async with self.mcp_client:
            for _ in range(self.max_iterations):
                response_message = self.llm(messages)
                messages.append(response_message)

                tool_calls = response_message.get("tool_calls")
                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    result = await self._execute_tool(tool_call)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )

        return messages


if __name__ == "__main__":
    qg = QueryGenerator()
