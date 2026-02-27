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

from openllm_func_call_synthesizer.core import critic as critic_module


@pytest.fixture()
def critic(monkeypatch):
    """Return a Critic instance without invoking the real curator backend."""

    monkeypatch.setattr(critic_module.curator.LLM, "__init__", lambda self, *args, **kwargs: None)
    return critic_module.Critic(model_name="test-model")


def build_payload(**overrides) -> dict:
    payload = {
        "task_prompt": "Plan the best action",
        "query": "How do I tidy up?",
        "functions": [{"name": "organize", "description": "Tidy a room"}],
        "label": "",
        "response": json.dumps({
            "content": '<think>I should reason carefully</think>{"result":"OK"}'
        }, ensure_ascii=False),
    }
    payload.update(overrides)
    return payload


def test_prompt_requires_task_prompt(critic):
    payload = build_payload(task_prompt="")

    with pytest.raises(ValueError):
        critic.prompt(payload)


def test_prompt_allows_label_without_answer(critic):
    payload = build_payload(label="Preferred response", response="")

    prompt_messages = critic.prompt(payload)

    assert isinstance(prompt_messages, list)
    assert any("Preferred response" in message["content"] for message in prompt_messages if "content" in message)


def test_parse_filters_think_and_extracts_json(critic):
    payload = build_payload()
    response = {
        "choices": [
            {
                "message": {
                    "content": "<think>Internal chain</think>{\"score\": 4, \"reason\": \"Accurate\"}"
                }
            }
        ]
    }

    parsed = critic.parse(payload, response)

    assert parsed["score"] == 4
    assert parsed["reason"] == "Accurate"
    assert parsed["raw_critic_output"].startswith("<think>")
    assert parsed["prompt"][0]["role"] == "system"
