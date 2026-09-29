import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Groq's free tier allows only ~8,000 tokens per minute for gpt-oss-120b.
# Tool output is fed back into the model on every step, so keep it small.
# (Roughly 4 characters ~ 1 token.)
MAX_SEARCH_CHARS = 3500
MAX_SCRAPE_CHARS = 3000
SEARCH_RESULTS = 4


def _truncate(result, limit):
    text = result if isinstance(result, str) else str(result)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated to save tokens]"


class LimitedSearchTool(SerperDevTool):
    def _run(self, *args, **kwargs):
        return _truncate(super()._run(*args, **kwargs), MAX_SEARCH_CHARS)


class LimitedScrapeTool(ScrapeWebsiteTool):
    def _run(self, *args, **kwargs):
        return _truncate(super()._run(*args, **kwargs), MAX_SCRAPE_CHARS)


def get_search_tool():
    if not os.getenv("SERPER_API_KEY"):
        raise RuntimeError(
            "SERPER_API_KEY is not configured. "
            "Create a Serper API key and add it to Streamlit Secrets."
        )
    return LimitedSearchTool(n_results=SEARCH_RESULTS)


def get_scrape_tool():
    return LimitedScrapeTool()


def get_research_tools():
    return [get_search_tool(), get_scrape_tool()]
