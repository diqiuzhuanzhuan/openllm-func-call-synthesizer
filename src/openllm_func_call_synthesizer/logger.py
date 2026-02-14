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

"""Logging utilities for openllm_func_call_synthesizer."""

from __future__ import annotations

import logging
import os

_DEFAULT_LOG_LEVEL = os.getenv("OPENLLM_FUNC_CALL_SYNTHESIZER_LOG_LEVEL", "INFO")
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_level(level: object) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    return logging.INFO


def _ensure_configured(logger: logging.Logger) -> logging.Logger:
    if getattr(logger, "_openllm_logger_configured", False):
        return logger

    level = _parse_level(_DEFAULT_LOG_LEVEL)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    logger._openllm_logger_configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger namespaced to the package."""
    base_name = "openllm_func_call_synthesizer"
    full_name = base_name if not name else f"{base_name}.{name}"
    return _ensure_configured(logging.getLogger(full_name))


logger = get_logger()
