import os
import re
import time
import asyncio

import litellm
from crewai import LLM

MODEL_NAME = "groq/openai/gpt-oss-120b"

# Retry settings for Groq rate limits (tokens-per-minute on the free tier)
RATE_LIMIT_MAX_RETRIES = 8
RATE_LIMIT_DEFAULT_WAIT = 10   # seconds, used if Groq doesn't say how long to wait
RATE_LIMIT_MAX_WAIT = 60       # never wait longer than this between retries

# Retry settings for Groq "tool_use_failed" errors
TOOL_ERROR_MAX_RETRIES = 2

FINAL_ANSWER_NUDGE = (
    "Do not call any tools. Write your final answer now, using only the "
    "information gathered so far."
)


# --- Workaround 1: CrewAI bug, `cache_breakpoint` leaking to non-Anthropic providers ---
# CrewAI adds an internal "cache_breakpoint" key to messages (used for Anthropic
# prompt caching). Groq rejects unknown message fields, so strip it before sending.

def _strip_cache_breakpoint(messages):
    if not messages:
        return messages
    return [
        {k: v for k, v in m.items() if k != "cache_breakpoint"}
        if isinstance(m, dict)
        else m
        for m in messages
    ]


# --- Workaround 2: wait and retry when Groq's rate limit is hit ---
# Groq's error says e.g. "Please try again in 3.735s". Parse it, sleep, retry.

def _wait_time_from_error(error):
    match = re.search(r"try again in ([\d.]+)\s*(ms|s)", str(error))
    if match:
        value = float(match.group(1))
        seconds = value / 1000 if match.group(2) == "ms" else value
        return min(seconds + 2, RATE_LIMIT_MAX_WAIT)  # +2s safety buffer
    return RATE_LIMIT_DEFAULT_WAIT


# --- Workaround 3: recover from Groq "tool_use_failed" errors ---
# When an agent hits its iteration limit, CrewAI forces a final answer with
# tool_choice="none", but gpt-oss may still try to call a tool and Groq rejects
# it ("Tool choice is none, but model called a tool"). In that case we retry
# without tools and explicitly ask for a plain-text final answer. Other
# tool_use_failed errors (malformed tool calls) are simply retried.

def _is_tool_use_failed(error):
    text = str(error)
    return "tool_use_failed" in text or "Tool choice is none" in text


def _fix_kwargs_after_tool_error(kwargs, error):
    if "Tool choice is none" not in str(error):
        return kwargs
    fixed = dict(kwargs)
    fixed.pop("tools", None)
    fixed.pop("tool_choice", None)
    fixed["messages"] = list(fixed.get("messages") or []) + [
        {"role": "user", "content": FINAL_ANSWER_NUDGE}
    ]
    return fixed


if not getattr(litellm, "_crew_patched", False):
    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        rate_retries = 0
        tool_retries = 0
        while True:
            try:
                return _orig_completion(*args, **kwargs)
            except litellm.RateLimitError as e:
                if rate_retries >= RATE_LIMIT_MAX_RETRIES:
                    raise
                rate_retries += 1
                time.sleep(_wait_time_from_error(e))
            except litellm.BadRequestError as e:
                if not _is_tool_use_failed(e) or tool_retries >= TOOL_ERROR_MAX_RETRIES:
                    raise
                tool_retries += 1
                kwargs = _fix_kwargs_after_tool_error(kwargs, e)

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        rate_retries = 0
        tool_retries = 0
        while True:
            try:
                return await _orig_acompletion(*args, **kwargs)
            except litellm.RateLimitError as e:
                if rate_retries >= RATE_LIMIT_MAX_RETRIES:
                    raise
                rate_retries += 1
                await asyncio.sleep(_wait_time_from_error(e))
            except litellm.BadRequestError as e:
                if not _is_tool_use_failed(e) or tool_retries >= TOOL_ERROR_MAX_RETRIES:
                    raise
                tool_retries += 1
                kwargs = _fix_kwargs_after_tool_error(kwargs, e)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
    litellm._crew_patched = True
# ------------------------------------------------------------------------------------


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return LLM(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=1500,
    )
