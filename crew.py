from crewai import Crew, Process, Task

from agents.research_planner import create_research_planner
from agents.web_researcher import create_web_researcher
from agents.fact_checker import create_fact_checker
from agents.research_writer import create_research_writer

def run_research(user_query: str):
    planner = create_research_planner()
    researcher = create_web_researcher()
    checker = create_fact_checker()
    writer = create_research_writer()

    planning_task = Task(
        description=f"""
        Research question:
        {user_query}

        Create a practical research plan. Break the question into 4-7 specific
        sub-questions. Use your research tools to inspect the topic and identify
        the kinds of sources that should be consulted. Do not write the final
        report yet.
        """,
        expected_output=(
            "A structured research plan containing the main question, "
            "sub-questions, evidence requirements, and suggested source types."
        ),
        agent=planner,
    )

    research_task = Task(
        description=f"""
        Research question:
        {user_query}

        Follow the research plan produced by the Research Planner. Use web
        search and website extraction tools extensively. Collect evidence for
        each major sub-question.

        For each important finding, capture:
        - claim/finding
        - supporting evidence
        - source title or organization
        - source URL
        - publication date when available
        - whether the source is primary, secondary, or opinion

        Prefer primary and authoritative sources.
        """,
        expected_output=(
            "A source-backed research dossier organized by sub-question, "
            "including source URLs and evidence."
        ),
        agent=researcher,
        context=[planning_task],
    )

    checking_task = Task(
        description=f"""
        Audit the research dossier for this question:

        {user_query}

        Use your own search and scraping tools to independently verify the
        most important claims. Look specifically for:
        - unsupported claims
        - outdated information
        - conflicting figures
        - weak sources
        - claims repeated across sources without independent confirmation

        Produce a verification table with claim, verification status, evidence,
        source URL, and caveat where relevant.
        """,
        expected_output=(
            "A fact-checking and source-audit report with verified claims, "
            "disputed claims, unsupported claims, and important caveats."
        ),
        agent=checker,
        context=[planning_task, research_task],
    )

    writing_task = Task(
        description=f"""
        Write the final research report for:

        {user_query}

        Use the research dossier and fact-checking report. You may use your
        research tools to resolve any remaining uncertainty before writing.

        Required structure:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Areas of Agreement / Disagreement in the evidence
        5. Limitations and Uncertainty
        6. Sources

        Every important factual claim should be traceable to a source URL.
        Do not invent citations, statistics, dates, or quotes.
        """,
        expected_output=(
            "A polished, evidence-based research report with source URLs and "
            "clear caveats."
        ),
        agent=writer,
        context=[research_task, checking_task],
    )

    crew = Crew(
        agents=[planner, researcher, checker, writer],
        tasks=[planning_task, research_task, checking_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()
