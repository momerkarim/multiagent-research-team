import os
from crewai import LLM

MODEL_NAME = "groq/openai/gpt-oss-120b"

def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return LLM(
        model=MODEL_NAME,
        temperature=0.2,
        max_tokens=4096,
    )
