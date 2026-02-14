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

from bespokelabs import curator

from openllm_func_call_synthesizer.core.formatter import (
    CRITIC_FUNCTION_CALL_SYSTEM_HEADER,
)
from openllm_func_call_synthesizer.utils import extract_format


class Critic(curator.LLM):
    """A simple critic for any tasks."""

    return_completions_object = True

    #    def _hash_fingerprint(self, dataset_hash: str = "", disable_cache: bool = False):
    #        return super()._hash_fingerprint("", disable_cache)

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

    def prompt(self, input: dict) -> dict:
        """The prompt is used to generate the function call."""
        # Prepare a readable listing of available functions

        task_prompt = input.get(self.task_prompt_field, "")
        if not task_prompt:
            raise ValueError("task_prompt is required")
        query = input.get(self.query_field, "")
        if not query:
            raise ValueError("query is required")
        functions = input.get(self.functions_field, "")
        if not functions:
            raise ValueError("functions is required")
        if isinstance(functions, str):
            functions = json.dumps(json.loads(functions), ensure_ascii=False, indent=2)
        label = input.get(self.label_field, "")
        answer = input.get(self.response_field, "")
        if not label and not answer:
            raise ValueError("either label or answer is required")

        answer_data = json.loads(answer)
        content = answer_data.get("content")
        if content:
            answer_filter_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        else:
            answer_filter_think = answer

        print("===========answer_filter_think============\n", answer_filter_think)
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

    def parse(self, input: dict, response) -> dict:
        """Parse the response to extract the function call or the message."""
        input["prompt"] = self.prompt(input)

        # Clean the response content first
        raw_content = response["choices"][0]["message"]["content"]
        # Remove <think>...</think> blocks from the critic's own output if present
        clean_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        json_extract = extract_format(format="json", content=clean_content)
        if json_extract is None:
            # Fallback: try to extract from raw_content just in case regex failed on cleaned version
            # (e.g. if the json strictly relies on structure that was partly inside think - unlikely but possible)
            json_extract = extract_format(format="json", content=raw_content)

        if json_extract is None:
            input["score"] = 0
            input["reason"] = "Failed to parse critic response as JSON"
            # Optional: store raw output for debugging
            input["raw_critic_output"] = raw_content
        else:
            score, reason = json_extract.get("score", 0), json_extract.get("reason", "No reason provided")
            input["score"] = score
            input["reason"] = reason
            input["raw_critic_output"] = raw_content
        return input
