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
from deepeval.models.base_model import DeepEvalBaseLLM

from openllm_func_call_synthesizer.core.critic import DeepEvalCritic


class StubDeepEvalJudge(DeepEvalBaseLLM):
    """Real DeepEval model adapter used to exercise GEval end-to-end without network."""

    def load_model(self):
        return self

    def get_model_name(self):
        return "stub-deepeval-judge"

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            schema_name = getattr(schema, "__name__", "")
            if schema_name == "Steps":
                return schema.model_validate({"steps": ["Check function choice", "Check arguments"]})
            if schema_name == "ReasonScore":
                return schema.model_validate(
                    {
                        "score": 9,
                        "reason": "The selected function and extracted arguments are correct.",
                    }
                )
        return json.dumps(
            {
                "score": 9,
                "reason": "The selected function and extracted arguments are correct.",
            },
            ensure_ascii=False,
        )

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)


def build_payload(**overrides):
    payload = {
        "task_prompt": "Choose the right tool call for the request.",
        "query": "Play Love Story on the speaker.",
        "functions": [
            {
                "name": "play_media",
                "description": "Play media on a device.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "media_type": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["media_type", "title"],
                },
            }
        ],
        "label": '[{"name":"play_media","arguments":{"media_type":"music","title":"Love Story"}}]',
        "response": "",
    }
    payload.update(overrides)
    return payload


def test_deepeval_critic_scores_with_real_geval():
    critic = DeepEvalCritic(model_name="judge-model", backend="openai")
    critic.eval_model = StubDeepEvalJudge()

    result = critic.score_row(build_payload())

    assert result["score"] == pytest.approx(0.9)
    assert "selected function" in result["reason"]
    assert json.loads(result["raw_critic_output"])["metric"] == "GEval"
    assert result["critic_backend"] == "deepeval"


def test_litellm_deepeval_model_supports_schema(monkeypatch):
    from openllm_func_call_synthesizer.core.critic import LiteLLMDeepEvalModel

    class Message:
        content = '{"score": 0.75, "reason": "Structured output works."}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("openllm_func_call_synthesizer.core.critic.litellm.completion", lambda **kwargs: Response())

    model = LiteLLMDeepEvalModel(model_name="judge-model")

    class Schema:
        @classmethod
        def model_validate_json(cls, payload):
            data = json.loads(payload)
            return type("Result", (), data)

    result = model.generate("judge this", schema=Schema)
    assert result.score == 0.75
    assert result.reason == "Structured output works."
