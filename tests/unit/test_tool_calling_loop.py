# MIT License
#
# Copyright (c) 2025 LoongMa
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

import pytest

from openllm_func_call_synthesizer.core.synthesizer import ToolCallingLoop

# ---------------------------------------------------------------------------
# MCP client / result fakes
# ---------------------------------------------------------------------------


class FakeContent:
    """Minimal stand-in for mcp.types.TextContent."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeCallToolResult:
    """Minimal stand-in for fastmcp.client.CallToolResult."""

    def __init__(self, text: str, is_error: bool = False) -> None:
        self.content = [FakeContent(text)]
        self.is_error = is_error
        self.data = None


class FakeMCPClient:
    """Async context manager that records calls and returns scripted results.

    ``tool_results`` maps tool name → string result (or an Exception to raise).
    """

    def __init__(self, tool_results: dict) -> None:
        self.tool_results = tool_results
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def call_tool(self, name: str, arguments: dict, *, raise_on_error: bool = True):
        self.calls.append({"name": name, "arguments": arguments})
        outcome = self.tool_results.get(name)
        if isinstance(outcome, Exception):
            raise outcome
        text = str(outcome) if outcome is not None else f"result from {name}"
        return FakeCallToolResult(text)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def make_tool_call(tc_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def assistant(content: str | None = None, tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant"}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def seq_llm(*responses):
    """Return a sync callable that yields responses in order."""
    it = iter(responses)
    return lambda _: next(it)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_tool_calls_returns_immediately():
    """Loop stops after one LLM call when the assistant has no tool_calls."""
    call_count = 0

    def llm(messages):
        nonlocal call_count
        call_count += 1
        return assistant(content="Paris")

    result = await ToolCallingLoop(
        llm_callable=llm,
        mcp_client=FakeMCPClient({}),
    ).run(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Capital of France?"},
        ]
    )

    assert call_count == 1
    assert result[-1] == {"role": "assistant", "content": "Paris"}
    assert len(result) == 3  # system + user + assistant


@pytest.mark.anyio
async def test_single_tool_call_executes_and_continues():
    """A tool call is dispatched via MCP, result appended, then LLM called again."""
    client = FakeMCPClient({"get_weather": "Tokyo: sunny, 22°C"})
    llm = seq_llm(
        assistant(tool_calls=[make_tool_call("c1", "get_weather", {"city": "Tokyo"})]),
        assistant(content="The weather in Tokyo is sunny."),
    )

    result = await ToolCallingLoop(llm_callable=llm, mcp_client=client).run(
        [
            {"role": "system", "content": "Weather assistant."},
            {"role": "user", "content": "Weather in Tokyo?"},
        ]
    )

    # system + user + assistant(tool_calls) + tool + assistant(final)
    assert len(result) == 5
    tool_msg = result[3]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "c1"
    assert "Tokyo" in tool_msg["content"]
    assert result[-1]["content"] == "The weather in Tokyo is sunny."

    # Verify MCP was called with correct arguments
    assert client.calls == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


@pytest.mark.anyio
async def test_multiple_tool_calls_in_one_message_all_executed():
    """All tool calls in one assistant message are dispatched to MCP in order."""
    client = FakeMCPClient({"add": "3", "multiply": "12"})
    llm = seq_llm(
        assistant(
            tool_calls=[
                make_tool_call("ca", "add", {"x": 1, "y": 2}),
                make_tool_call("cb", "multiply", {"x": 3, "y": 4}),
            ]
        ),
        assistant(content="Results: 3 and 12"),
    )

    result = await ToolCallingLoop(llm_callable=llm, mcp_client=client).run(
        [{"role": "user", "content": "Add 1+2 and multiply 3*4"}]
    )

    # user + assistant(2 tool_calls) + tool_add + tool_multiply + assistant(final)
    assert len(result) == 5
    assert result[2]["role"] == "tool"
    assert result[2]["tool_call_id"] == "ca"
    assert result[2]["content"] == "3"
    assert result[3]["role"] == "tool"
    assert result[3]["tool_call_id"] == "cb"
    assert result[3]["content"] == "12"

    # MCP calls must preserve ordering
    assert [c["name"] for c in client.calls] == ["add", "multiply"]


@pytest.mark.anyio
async def test_tool_call_id_preserved_in_tool_message():
    """The tool message's tool_call_id must exactly match the id from tool_calls."""
    client = FakeMCPClient({"greet": "Hello, Alice!"})
    llm = seq_llm(
        assistant(tool_calls=[make_tool_call("unique_xyz", "greet", {"name": "Alice"})]),
        assistant(content="Done"),
    )

    result = await ToolCallingLoop(llm_callable=llm, mcp_client=client).run(
        [{"role": "user", "content": "Greet Alice"}]
    )

    tool_msg = next(m for m in result if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "unique_xyz"
    assert tool_msg["content"] == "Hello, Alice!"


@pytest.mark.anyio
async def test_mcp_exception_produces_error_content():
    """If the MCP client raises, the error is captured in the tool message content."""
    client = FakeMCPClient({"boom": RuntimeError("server exploded")})
    llm = seq_llm(
        assistant(tool_calls=[make_tool_call("c1", "boom", {})]),
        assistant(content="I couldn't do that"),
    )

    result = await ToolCallingLoop(llm_callable=llm, mcp_client=client).run(
        [{"role": "user", "content": "Trigger error"}]
    )

    tool_msg = next(m for m in result if m["role"] == "tool")
    assert "Error calling tool" in tool_msg["content"]
    assert "server exploded" in tool_msg["content"]


@pytest.mark.anyio
async def test_mcp_error_flag_in_result_prefixes_error():
    """When CallToolResult.is_error is True the content is prefixed with 'Error:'."""
    class ErrorClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def call_tool(self, name, arguments, *, raise_on_error=True):
            return FakeCallToolResult("something went wrong", is_error=True)

    llm = seq_llm(
        assistant(tool_calls=[make_tool_call("c1", "failing_tool", {})]),
        assistant(content="Could not complete"),
    )

    result = await ToolCallingLoop(llm_callable=llm, mcp_client=ErrorClient()).run(
        [{"role": "user", "content": "Use failing tool"}]
    )

    tool_msg = next(m for m in result if m["role"] == "tool")
    assert tool_msg["content"].startswith("Error:")
    assert "something went wrong" in tool_msg["content"]


@pytest.mark.anyio
async def test_max_iterations_stops_loop():
    """Loop terminates after max_iterations even if the LLM keeps returning tool_calls."""
    client = FakeMCPClient({"noop": "ok"})

    def llm(messages):
        return assistant(tool_calls=[make_tool_call("c", "noop", {})])

    result = await ToolCallingLoop(
        llm_callable=llm, mcp_client=client, max_iterations=3
    ).run([{"role": "user", "content": "Loop forever"}])

    assert len([m for m in result if m["role"] == "assistant"]) == 3


def test_invalid_max_iterations_raises():
    with pytest.raises(ValueError):
        ToolCallingLoop(llm_callable=lambda m: {}, mcp_client=None, max_iterations=0)


@pytest.mark.anyio
async def test_tool_arguments_forwarded_to_mcp():
    """Arguments from the tool_call are parsed and forwarded to MCP call_tool."""
    client = FakeMCPClient({"get_weather": "22°C"})
    llm = seq_llm(
        assistant(
            tool_calls=[make_tool_call("c1", "get_weather", {"location": "Berlin", "units": "celsius"})]
        ),
        assistant(content="22°C in Berlin"),
    )

    await ToolCallingLoop(llm_callable=llm, mcp_client=client).run(
        [{"role": "user", "content": "Weather in Berlin?"}]
    )

    assert client.calls[0]["arguments"] == {"location": "Berlin", "units": "celsius"}


@pytest.mark.anyio
async def test_multi_turn_tool_calls():
    """The loop handles multiple consecutive rounds of MCP tool calls correctly."""
    client = FakeMCPClient({"step1": "step1 result", "step2": "step2 result"})
    llm = seq_llm(
        assistant(tool_calls=[make_tool_call("r1", "step1", {})]),
        assistant(tool_calls=[make_tool_call("r2", "step2", {})]),
        assistant(content="All done"),
    )

    result = await ToolCallingLoop(
        llm_callable=llm, mcp_client=client, max_iterations=5
    ).run([{"role": "user", "content": "Run steps"}])

    roles = [m["role"] for m in result]
    # user → assistant(tool) → tool → assistant(tool) → tool → assistant(final)
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert result[-1]["content"] == "All done"
    assert [c["name"] for c in client.calls] == ["step1", "step2"]


@pytest.mark.anyio
async def test_seed_messages_not_mutated():
    """The initial messages list passed to run() must not be mutated."""
    initial = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    original_copy = list(initial)

    await ToolCallingLoop(
        llm_callable=lambda m: assistant(content="hello"),
        mcp_client=FakeMCPClient({}),
    ).run(initial)

    assert initial == original_copy


@pytest.mark.anyio
async def test_mcp_client_opened_once_per_run():
    """The async context manager on the MCP client is entered exactly once per run."""
    enter_count = 0

    class CountingClient(FakeMCPClient):
        async def __aenter__(self):
            nonlocal enter_count
            enter_count += 1
            return self

    client = CountingClient({"noop": "ok"})
    llm = seq_llm(
        assistant(tool_calls=[make_tool_call("c1", "noop", {})]),
        assistant(tool_calls=[make_tool_call("c2", "noop", {})]),
        assistant(content="done"),
    )

    await ToolCallingLoop(llm_callable=llm, mcp_client=client).run(
        [{"role": "user", "content": "go"}]
    )

    assert enter_count == 1
