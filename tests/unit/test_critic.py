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
def legacy_critic(monkeypatch):
    """Return a LegacyCritic instance without invoking the real curator backend."""

    monkeypatch.setattr(critic_module.curator.LLM, "__init__", lambda self, *args, **kwargs: None)
    return critic_module.LegacyCritic(model_name="test-model")


def build_payload(**overrides) -> dict:
    payload = {
        "task_prompt": "Plan the best action",
        "query": "How do I tidy up?",
        "functions": [{"name": "organize", "description": "Tidy a room"}],
        "label": "",
        "response": json.dumps(
            {
                "content": '<think>I should reason carefully</think>{"result":"OK"}',
            },
            ensure_ascii=False,
        ),
    }
    payload.update(overrides)
    return payload


def test_prompt_requires_task_prompt(legacy_critic):
    payload = build_payload(task_prompt="")

    with pytest.raises(ValueError):
        legacy_critic.prompt(payload)


def test_prompt_allows_label_without_answer(legacy_critic):
    payload = build_payload(label="Preferred response", response="")

    prompt_messages = legacy_critic.prompt(payload)

    assert isinstance(prompt_messages, list)
    assert any("Preferred response" in message["content"] for message in prompt_messages if "content" in message)


def test_parse_filters_think_and_extracts_json(legacy_critic):
    payload = build_payload()
    response = {
        "choices": [
            {
                "message": {
                    "content": '<think>Internal chain</think>{"score": 4, "reason": "Accurate"}',
                }
            }
        ]
    }

    parsed = legacy_critic.parse(payload, response)

    assert parsed["score"] == 4
    assert parsed["reason"] == "Accurate"
    assert parsed["raw_critic_output"].startswith("<think>")
    assert parsed["prompt"][0]["role"] == "system"


class FakeParams:
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    CONTEXT = "context"
    EXPECTED_OUTPUT = "expected_output"


class FakeLLMTestCase:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeGEval:
    last_init = None
    last_case = None

    def __init__(self, **kwargs) -> None:
        FakeGEval.last_init = kwargs
        self.score = 0.82
        self.reason = "The function choice and arguments are correct."

    def measure(self, test_case) -> None:
        FakeGEval.last_case = test_case


@pytest.fixture()
def deepeval_critic(monkeypatch):
    monkeypatch.setattr(critic_module, "GEval", FakeGEval)
    monkeypatch.setattr(critic_module, "LLMTestCase", FakeLLMTestCase)
    monkeypatch.setattr(critic_module, "DeepEvalParams", FakeParams)
    return critic_module.DeepEvalCritic(model_name="judge-model", backend="openai")


def test_deepeval_critic_uses_label_when_present(deepeval_critic):
    row = build_payload(label='[{"name":"organize","arguments":{"room":"bedroom"}}]', response="")

    scored = deepeval_critic.score_row(row)

    assert scored["score"] == pytest.approx(0.82)
    assert scored["reason"] == "The function choice and arguments are correct."
    assert scored["critic_backend"] == "deepeval"
    assert json.loads(scored["raw_critic_output"])["backend"] == "deepeval"
    assert FakeGEval.last_case is not None
    assert FakeGEval.last_case.kwargs["actual_output"] == row["label"]
    assert "Available functions" in FakeGEval.last_case.kwargs["context"][1]


def test_deepeval_critic_falls_back_to_answer_content(deepeval_critic):
    row = build_payload(
        label="",
        response=json.dumps({"content": '<think>scratch</think>{"tool":"organize"}'}, ensure_ascii=False),
    )

    deepeval_critic.score_row(row)

    assert FakeGEval.last_case is not None
    assert FakeGEval.last_case.kwargs["actual_output"] == '{"tool":"organize"}'


def test_deepeval_critic_requires_ground_truth_when_enabled(monkeypatch):
    monkeypatch.setattr(critic_module, "GEval", FakeGEval)
    monkeypatch.setattr(critic_module, "LLMTestCase", FakeLLMTestCase)
    monkeypatch.setattr(critic_module, "DeepEvalParams", FakeParams)
    critic = critic_module.DeepEvalCritic(model_name="judge-model", backend="openai", use_gt=True)

    with pytest.raises(ValueError):
        critic.score_row(build_payload())


def test_build_critic_selects_backends(monkeypatch):
    monkeypatch.setattr(critic_module.curator.LLM, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(critic_module, "GEval", FakeGEval)
    monkeypatch.setattr(critic_module, "LLMTestCase", FakeLLMTestCase)
    monkeypatch.setattr(critic_module, "DeepEvalParams", FakeParams)

    legacy = critic_module.build_critic(backend_type="legacy_llm", model_name="legacy")
    modern = critic_module.build_critic(backend_type="deepeval", model_name="judge")

    assert isinstance(legacy, critic_module.LegacyCritic)
    assert isinstance(modern, critic_module.DeepEvalCritic)
