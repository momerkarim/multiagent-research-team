from crewai import Agent
from config import get_llm
from tools import get_research_tools

def create_research_planner():
    return Agent(
        role="Research Planner",
        goal=(
            "Turn the user's research question into a rigorous research plan, "
            "identify the key questions that must be answered, and identify "
            "what evidence is required."
        ),
        backstory=(
            "You are a senior research strategist. You think in hypotheses, "
            "sub-questions, evidence requirements, and source quality. "
            "You use the available research tools to inspect the topic before "
            "finalizing the research plan."
        ),
        tools=get_research_tools(),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
