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

import pytest

from openllm_func_call_synthesizer.core.synthesizer import ConversationGenerator


class FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def to_list(self):
        return self._rows


class FakeResponse:
    def __init__(self, next_turn):
        self.dataset = FakeDataset([{"next_turn": next_turn}])


class FakeLLM:
    def __init__(self, turns):
        self.turns = iter(turns)

    def __call__(self, rows):  # pragma: no cover - simple shim
        return FakeResponse(next(self.turns))


def test_generate_conversation_alternates_roles():
    generator = ConversationGenerator(
        user_model_name="human",
        assistant_model_name="assistant",
        max_turns=4,
        human_llm=FakeLLM([
        {"role": "user", "content": "Human message 1"},
        {"role": "user", "content": "Human message 2"},
        ]),
        assistant_llm=FakeLLM([
        {"role": "assistant", "content": "Assistant reply 1"},
        {"role": "assistant", "content": "Assistant reply 2"},
        ]),
    )

    conversation = generator.generate("Book a trip to Japan")

    assert len(conversation) == 4
    assert [turn["role"] for turn in conversation] == ["user", "assistant", "user", "assistant"]


def test_generate_conversation_respects_turn_limit():
    generator = ConversationGenerator(
        user_model_name="human",
        assistant_model_name="assistant",
        max_turns=3,
        human_llm=FakeLLM([
        {"role": "user", "content": "first human"},
        {"role": "user", "content": "second human"},
        ]),
        assistant_llm=FakeLLM([
        {"role": "assistant", "content": "assistant one"},
        ]),
    )

    conversation = generator.generate("Discuss recipes")

    assert len(conversation) == 3
    assert conversation[-1]["role"] == "user"


def test_invalid_turn_limit_raises():
    with pytest.raises(ValueError):
        ConversationGenerator(user_model_name="u", assistant_model_name="a", max_turns=1)
