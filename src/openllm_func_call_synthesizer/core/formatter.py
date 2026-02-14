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

from pathlib import Path

with (Path(__file__).parents[0] / Path("templates") / Path("critic_system_header.md")).open("r") as f:
    __BASE_CRITIC_SYSTEM_HEADER = f.read()

CRITIC_SYSTEM_HEADER = __BASE_CRITIC_SYSTEM_HEADER.replace("{context_prompt}", "", 1)


with (Path(__file__).parents[0] / Path("templates") / Path("critic_function_call_system_header.md")).open("r") as f:
    __BASE_CRITIC_FUNCTION_CALL_SYSTEM_HEADER = f.read()
    # noqa: E501

CRITIC_FUNCTION_CALL_SYSTEM_HEADER = __BASE_CRITIC_FUNCTION_CALL_SYSTEM_HEADER.replace("{context_prompt}", "", 1)

with (Path(__file__).parents[0] / Path("templates") / Path("query_generate_system_header.md")).open("r") as f:
    __BASE_QUERY_GENERATE_SYSTEM_HEADER = f.read()

QUERY_GENERATE_SYSTEM_HEADER = __BASE_QUERY_GENERATE_SYSTEM_HEADER
