import os

import litellm
from crewai import LLM

MODEL_NAME = "groq/openai/gpt-oss-120b"


# --- Workaround for CrewAI bug: `cache_breakpoint` leaking to non-Anthropic providers ---
# CrewAI adds an internal "cache_breakpoint" key to messages (used for Anthropic
# prompt caching). Groq rejects unknown message fields, so we strip it before
# the request is sent through LiteLLM.

def _strip_cache_breakpoint(messages):
    if not messages:
        return messages
    return [
        {k: v for k, v in m.items() if k != "cache_breakpoint"}
        if isinstance(m, dict)
        else m
        for m in messages
    ]


if not getattr(litellm, "_cache_breakpoint_patched", False):
    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        return _orig_completion(*args, **kwargs)

    async def _patched_acompletion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        return await _orig_acompletion(*args, **kwargs)

    litellm.completion = _patched_completion
    litellm.acompletion = _patched_acompletion
    litellm._cache_breakpoint_patched = True
# ------------------------------------------------------------------------------------


def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return LLM(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=4096,
    )
