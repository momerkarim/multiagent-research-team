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


# --- Workaround 2: automatically wait and retry when Groq's rate limit is hit ---
# Groq's error message says e.g. "Please try again in 3.735s". We parse that,
# sleep, and retry instead of crashing the whole crew.

def _wait_time_from_error(error):
    match = re.search(r"try again in ([\d.]+)\s*(ms|s)", str(error))
    if match:
        value = float(match.group(1))
        seconds = value / 1000 if match.group(2) == "ms" else value
        return min(seconds + 2, RATE_LIMIT_MAX_WAIT)  # +2s safety buffer
    return RATE_LIMIT_DEFAULT_WAIT


if not getattr(litellm, "_crew_patched", False):
    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return _orig_completion(*args, **kwargs)
            except litellm.RateLimitError as e:
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise
                time.sleep(_wait_time_from_error(e))

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                return await _orig_acompletion(*args, **kwargs)
            except litellm.RateLimitError as e:
                if attempt == RATE_LIMIT_MAX_RETRIES:
                    raise
                await asyncio.sleep(_wait_time_from_error(e))

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
