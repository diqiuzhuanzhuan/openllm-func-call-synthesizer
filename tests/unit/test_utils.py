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

from openllm_func_call_synthesizer.core.synthesizer import FunctionCallGenerator
from openllm_func_call_synthesizer.utils import extract_format, parse_hermes_tool_calls, pick_unique


def test_parse_hermes_tool_calls_from_openai_dict():
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "search_photos",
                    "arguments": json.dumps({"album": "夏日相册", "filters": {"color": "warm"}}),
                }
            }
        ]
    }

    result = parse_hermes_tool_calls(message)

    assert result == [
        {
            "name": "search_photos",
            "arguments": {"album": "夏日相册", "filters": {"color": "warm"}},
        }
    ]


def test_parse_hermes_tool_calls_from_string_block():
    message = (
        "<tool_call>\n"
        '{"name": "send_email", "arguments": {"to": "team@acme.ai", "body": "Ping"}}\n'
        "</tool_call>"
    )

    result = parse_hermes_tool_calls(message)

    assert result == [
        {
            "name": "send_email",
            "arguments": {"to": "team@acme.ai", "body": "Ping"},
        }
    ]


def test_deduplicate_input_removes_tool_call_ids():
    generator = FunctionCallGenerator.__new__(FunctionCallGenerator)

    def build_item(tool_call_id: str) -> dict:
        answer = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "function": {
                            "name": "create_album",
                            "arguments": json.dumps({"title": "Weekend"}),
                        },
                    }
                ]
            }
        )
        return {
            "prompt": [{"role": "user", "content": "Make an album"}],
            "function_call": json.dumps([
                {"name": "create_album", "arguments": {"title": "Weekend"}}
            ], ensure_ascii=False),
            "answer": answer,
        }

    deduped = generator._deduplicate_input_ls([build_item("call_1"), build_item("call_2")])

    assert len(deduped) == 1
    assert deduped[0]["function_call"].count("Weekend") == 1


def test_extract_format_reads_code_block():
    content = """
    ```json
    {"variations": [{"query": "hello", "dimension": "baseline"}]}
    ```
    """

    parsed = extract_format(format="json", content=content)

    assert parsed == {"variations": [{"query": "hello", "dimension": "baseline"}]}


def test_pick_unique_from_list_of_dicts():
    dataset = [
        {"query": "hello", "function": "foo"},
        {"query": "hello", "function": "bar"},
        {"query": "hi", "function": "baz"},
    ]

    result = pick_unique(dataset, field="query", k=2)

    assert len(result) == 2
    assert {item["query"] for item in result} == {"hello", "hi"}
