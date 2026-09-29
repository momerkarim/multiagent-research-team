from crewai import Agent
from config import get_llm
from tools import get_research_tools

def create_fact_checker():
    return Agent(
        role="Fact Checker and Source Auditor",
        goal=(
            "Audit the research findings, verify important claims against "
            "independent sources, identify contradictions, and flag weak or "
            "unsupported evidence."
        ),
        backstory=(
            "You are a skeptical research auditor. Never accept an important "
            "claim simply because another agent reported it. Search for "
            "independent confirmation, inspect original sources when possible, "
            "and clearly identify uncertainty."
        ),
        tools=get_research_tools(),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
