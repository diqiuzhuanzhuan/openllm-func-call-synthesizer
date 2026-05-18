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
import re
from typing import Any

import litellm
from bespokelabs import curator
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams as DeepEvalParams

from openllm_func_call_synthesizer.core.formatter import (
    CRITIC_FUNCTION_CALL_SYSTEM_HEADER,
)
from openllm_func_call_synthesizer.utils import extract_format


def _strip_think_blocks(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _normalize_functions(functions: Any) -> str:
    if isinstance(functions, str):
        return json.dumps(json.loads(functions), ensure_ascii=False, indent=2)
    return json.dumps(functions, ensure_ascii=False, indent=2)


class LegacyCritic(curator.LLM):
    """Legacy prompt-based critic powered by curator."""

    return_completions_object = True

    def __init__(
        self,
        model_name,
        response_format=None,
        batch=False,
        backend=None,
        generation_params=None,
        backend_params=None,
        system_prompt=None,
        query_field="query",
        task_prompt_field="task_prompt",
        label_field="label",
        functions_field="functions",
        response_field="response",
        purpose="function_call",
        use_gt=False,
        **kwargs,
    ):
        super().__init__(
            model_name, response_format, batch, backend, generation_params, backend_params, system_prompt, **kwargs
        )
        self.query_field = query_field
        self.task_prompt_field = task_prompt_field
        self.label_field = label_field
        self.functions_field = functions_field
        self.response_field = response_field
        self.use_gt = use_gt
        self.purpose = purpose

    def prompt(self, input: Any) -> Any:
        """The prompt is used to generate the function call."""
        input = dict(input)
        task_prompt = input.get(self.task_prompt_field, "")
        if not task_prompt:
            raise ValueError("task_prompt is required")
        query = input.get(self.query_field, "")
        if not query:
            raise ValueError("query is required")
        functions = input.get(self.functions_field, "")
        if not functions:
            raise ValueError("functions is required")
        functions = _normalize_functions(functions)
        label = input.get(self.label_field, "")
        answer = input.get(self.response_field, "")
        if not label and not answer:
            raise ValueError("either label or answer is required")

        answer_filter_think = ""
        if answer:
            answer_data = json.loads(answer)
            content = answer_data.get("content")
            if content:
                answer_filter_think = _strip_think_blocks(content)
            else:
                answer_filter_think = answer

        model_output = label if label else answer_filter_think
        ground_truth = input.get("ground_truth", "")

        if self.use_gt:
            if not ground_truth:
                raise ValueError("ground_truth is required")
            user_prompt = f"""
          The given instruction is {task_prompt}.
          The available functions are: {functions}.
          The model output is :{model_output}.
          The ground truth is:{ground_truth}.
          You need to score the model output based on the ground truth and the scoring rules,
          making sure not to confuse the model output with the ground truth.
          """
        else:
            user_prompt = f"""
          The given instruction is {task_prompt}.
          The available functions are: {functions}.
          The model output is :{model_output}.
          """
        return [
            {"role": "system", "content": CRITIC_FUNCTION_CALL_SYSTEM_HEADER},
            {"role": "user", "content": user_prompt},
        ]

    def parse(self, input: Any, response: Any) -> Any:
        """Parse the response to extract the function call or the message."""
        input = dict(input)
        input["prompt"] = self.prompt(input)

        raw_content = response["choices"][0]["message"]["content"]
        clean_content = _strip_think_blocks(raw_content)

        json_extract = extract_format(format="json", content=clean_content)
        if json_extract is None:
            json_extract = extract_format(format="json", content=raw_content)

        if json_extract is None:
            input["score"] = 0
            input["reason"] = "Failed to parse critic response as JSON"
            input["raw_critic_output"] = raw_content
        else:
            score, reason = json_extract.get("score", 0), json_extract.get("reason", "No reason provided")
            input["score"] = score
            input["reason"] = reason
            input["raw_critic_output"] = raw_content
        return input


class LiteLLMDeepEvalModel(DeepEvalBaseLLM):
    """DeepEval model wrapper that routes judge calls through litellm."""

    def __init__(
        self,
        model_name: str,
        *,
        backend: str | None = None,
        backend_params: dict | None = None,
        generation_params: dict | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.backend_params = backend_params or {}
        self.generation_params = generation_params or {}
        self.system_prompt = system_prompt or "You are a rigorous function calling evaluator."

    def load_model(self):
        return self

    def get_model_name(self):
        return self.model_name

    @staticmethod
    def _coerce_schema_output(content: str, schema):
        if schema is None:
            return content
        if hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(content)
        return schema(**json.loads(content))

    def _completion(self, prompt: str) -> str:
        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            **self.backend_params,
            **self.generation_params,
        )
        message = response.choices[0].message
        if hasattr(message, "content"):
            return message.content or ""
        if hasattr(message, "model_dump"):
            return message.model_dump().get("content", "")
        return ""

    def generate(self, prompt: str, schema=None):
        return self._coerce_schema_output(self._completion(prompt), schema)

    async def a_generate(self, prompt: str, schema=None):
        return self._coerce_schema_output(self._completion(prompt), schema)


class DeepEvalCritic:
    """DeepEval-backed critic that emits dataset-compatible score and reason fields."""

    def __init__(
        self,
        model_name: str,
        response_format=None,
        batch=False,
        backend=None,
        generation_params=None,
        backend_params=None,
        system_prompt=None,
        query_field="query",
        task_prompt_field="task_prompt",
        label_field="label",
        functions_field="functions",
        response_field="response",
        purpose="function_call",
        use_gt=False,
        threshold: float = 0.5,
        **kwargs,
    ) -> None:
        del response_format, batch, kwargs
        self.model_name = model_name
        self.backend = backend
        self.generation_params = generation_params or {}
        self.backend_params = backend_params or {}
        self.system_prompt = system_prompt
        self.query_field = query_field
        self.task_prompt_field = task_prompt_field
        self.label_field = label_field
        self.functions_field = functions_field
        self.response_field = response_field
        self.use_gt = use_gt
        self.purpose = purpose
        self.threshold = threshold
        self.eval_model = LiteLLMDeepEvalModel(
            model_name=model_name,
            backend=backend,
            backend_params=self.backend_params,
            generation_params=self.generation_params,
            system_prompt=system_prompt,
        )

    def _validate_input(self, row: dict) -> tuple[str, str, str, str, str]:
        task_prompt = row.get(self.task_prompt_field, "")
        if not task_prompt:
            raise ValueError("task_prompt is required")
        query = row.get(self.query_field, "")
        if not query:
            raise ValueError("query is required")
        functions = row.get(self.functions_field, "")
        if not functions:
            raise ValueError("functions is required")
        normalized_functions = _normalize_functions(functions)
        label = row.get(self.label_field, "")
        answer = row.get(self.response_field, "")
        if not label and not answer:
            raise ValueError("either label or answer is required")
        answer_filter_think = ""
        if answer:
            answer_data = json.loads(answer)
            content = answer_data.get("content")
            if content:
                answer_filter_think = _strip_think_blocks(content)
            else:
                answer_filter_think = answer
        model_output = label if label else answer_filter_think
        if self.use_gt and not row.get("ground_truth", ""):
            raise ValueError("ground_truth is required")
        return task_prompt, query, normalized_functions, model_output, answer_filter_think

    def _build_metric(self):
        metric_cls = GEval
        params_cls = DeepEvalParams
        assert metric_cls is not None
        assert params_cls is not None
        evaluation_params = [
            params_cls.INPUT,
            params_cls.ACTUAL_OUTPUT,
            params_cls.CONTEXT,
        ]
        if self.use_gt:
            evaluation_params.append(params_cls.EXPECTED_OUTPUT)
        return metric_cls(
            name="FunctionCallCorrectness",
            criteria=(
                "Evaluate whether the generated function call correctly satisfies the user's intent. "
                "Check function selection, required and optional arguments, schema compliance, argument types, "
                "and language consistency for free-text arguments. Return a higher score only when the output is "
                "fully usable as a function call."
            ),
            evaluation_steps=[
                "Check whether the selected function matches the user's request.",
                "Verify all required parameters are present and correctly extracted.",
                "Verify optional parameters are only included when supported by the request or context.",
                "Check that parameter values respect the declared schema and types.",
                "Penalize free-text argument language mismatches against the user input unless the value is an enum.",
            ],
            evaluation_params=evaluation_params,
            threshold=self.threshold,
            model=self.eval_model,
        )

    def _build_test_case(self, row: dict[str, Any]) -> Any:
        test_case_cls = LLMTestCase
        assert test_case_cls is not None
        task_prompt, query, functions, model_output, _ = self._validate_input(row)
        context = [
            f"Task prompt:\n{task_prompt}",
            f"Available functions:\n{functions}",
        ]
        if self.use_gt:
            return test_case_cls(
                input=query,
                actual_output=model_output,
                context=context,
                expected_output=str(row["ground_truth"]),
            )
        return test_case_cls(
            input=query,
            actual_output=model_output,
            context=context,
        )

    def score_row(self, row: dict) -> dict:
        metric = self._build_metric()
        test_case = self._build_test_case(row)
        metric.measure(test_case)

        reason = getattr(metric, "reason", "") or "No reason provided"
        score = float(getattr(metric, "score", 0.0) or 0.0)
        row["score"] = score
        row["reason"] = reason
        row["raw_critic_output"] = json.dumps(
            {
                "backend": "deepeval",
                "metric": getattr(metric, "__class__", type(metric)).__name__,
                "score": score,
                "reason": reason,
            },
            ensure_ascii=False,
        )
        row["critic_backend"] = "deepeval"
        return row

    def evaluate_dataset(self, dataset):
        return dataset.map(self.score_row)


def build_critic(*, backend_type: str = "legacy_llm", **kwargs):
    if backend_type == "legacy_llm":
        return LegacyCritic(**kwargs)
    if backend_type == "deepeval":
        return DeepEvalCritic(**kwargs)
    raise ValueError(f"Unsupported critic backend_type: {backend_type}")


def score_dataset_with_critic(critic, dataset):
    if isinstance(critic, LegacyCritic):
        return critic(dataset=dataset).dataset
    if hasattr(critic, "evaluate_dataset"):
        return critic.evaluate_dataset(dataset)
    raise TypeError(f"Unsupported critic instance: {type(critic)!r}")


Critic = LegacyCritic
