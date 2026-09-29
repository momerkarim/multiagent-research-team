# Multi-Agent Research Team

A beginner-friendly CrewAI research application using:

- Streamlit UI
- CrewAI multi-agent orchestration
- Groq `openai/gpt-oss-120b`
- Serper web search
- CrewAI website scraping tools

## Agents

1. Research Planner
2. Web Research Specialist
3. Fact Checker and Source Auditor
4. Senior Research Report Writer

## Recommended deployment runtime

Use Python 3.12 in Streamlit Community Cloud.

## Required secrets

Add these to Streamlit Community Cloud:

```toml
GROQ_API_KEY = "your-groq-key"
SERPER_API_KEY = "your-serper-key"
```

Never commit API keys to GitHub.

## Run

The app entrypoint is:

```text
app.py
```

## Architecture

User -> Streamlit -> CrewAI sequential crew

Planner -> Researcher -> Fact Checker -> Writer

The research agents use web search/scraping tools rather than relying only on the LLM's internal knowledge.
