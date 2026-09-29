import os
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

def get_search_tool():
    if not os.getenv("SERPER_API_KEY"):
        raise RuntimeError(
            "SERPER_API_KEY is not configured. "
            "Create a Serper API key and add it to Streamlit Secrets."
        )
    return SerperDevTool()

def get_scrape_tool():
    return ScrapeWebsiteTool()

def get_research_tools():
    return [get_search_tool(), get_scrape_tool()]
