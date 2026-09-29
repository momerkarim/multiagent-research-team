from crewai import Agent
from config import get_llm
from tools import get_research_tools

def create_research_writer():
    return Agent(
        role="Senior Research Report Writer",
        goal=(
            "Transform the verified research into a clear, structured, "
            "evidence-based report with source references and explicit "
            "uncertainty where evidence is incomplete."
        ),
        backstory=(
            "You are a senior analyst and research writer. You synthesize "
            "multiple sources without inventing facts. You prioritize primary "
            "sources, preserve important caveats, and produce a report that "
            "a decision-maker can understand quickly."
        ),
        tools=get_research_tools(),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )
