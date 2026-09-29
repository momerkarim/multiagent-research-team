from crewai import Agent
from config import get_llm
from tools import get_research_tools

def create_web_researcher():
    return Agent(
        role="Web Research Specialist",
        goal=(
            "Collect high-quality, current evidence for every major research "
            "question, using web search and direct page extraction."
        ),
        backstory=(
            "You are an investigative web researcher. You search broadly, "
            "open relevant sources, extract useful evidence, record source "
            "names and URLs, and distinguish facts from opinions or marketing."
        ),
        tools=get_research_tools(),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
