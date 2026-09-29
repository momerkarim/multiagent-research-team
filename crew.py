from crewai import Crew, Process, Task

from agents.research_planner import create_research_planner
from agents.web_researcher import create_web_researcher
from agents.fact_checker import create_fact_checker
from agents.research_writer import create_research_writer

# Groq free tier: ~8,000 tokens/minute for gpt-oss-120b.
# These settings keep each step small and space out the LLM calls.
MAX_RPM = 3          # max LLM requests per minute for the whole crew
AGENT_MAX_ITER = 8   # max reasoning/tool steps per agent per task


def run_research(user_query: str):
    planner = create_research_planner()
    researcher = create_web_researcher()
    checker = create_fact_checker()
    writer = create_research_writer()

    # Limit how many steps each agent may loop through.
    for agent in (planner, researcher, checker, writer):
        try:
            agent.max_iter = AGENT_MAX_ITER
        except Exception:
            pass

    planning_task = Task(
        description=f"""
        Research question:
        {user_query}

        Create a short research plan. Break the question into 3-4 specific
        sub-questions and list the types of sources to consult.
        Use at most 1 search. Do not write the final report.
        Keep the plan under 200 words.
        """,
        expected_output=(
            "A brief research plan (under 200 words) with the main question, "
            "3-4 sub-questions, and suggested source types."
        ),
        agent=planner,
    )

    research_task = Task(
        description=f"""
        Research question:
        {user_query}

        Follow the research plan. Use at most 3 searches and 2 page scrapes in
        total. Prefer primary and authoritative sources.

        For each key finding (max 6), record in one or two lines:
        - the finding
        - source name and URL
        - publication date if available
        Keep the whole dossier under 500 words.
        """,
        expected_output=(
            "A concise source-backed dossier (under 500 words) listing key "
            "findings with source URLs."
        ),
        agent=researcher,
        context=[planning_task],
    )

    checking_task = Task(
        description=f"""
        Audit the research dossier for this question:

        {user_query}

        Verify only the 3 most important claims. Use at most 2 searches and
        1 page scrape in total. Look for unsupported, outdated, or conflicting
        claims and weak sources.

        Produce a short table: claim | status (verified/disputed/unsupported)
        | source URL | caveat. Keep it under 300 words.
        """,
        expected_output=(
            "A short fact-check table (under 300 words) with claim status, "
            "source URL, and caveats."
        ),
        agent=checker,
        context=[research_task],
    )

    writing_task = Task(
        description=f"""
        Write the final research report for:

        {user_query}

        Use the research dossier and fact-check report. Do NOT use research
        tools unless absolutely necessary. Keep the report under 700 words.

        Required structure:
        1. Executive Summary
        2. Key Findings
        3. Detailed Analysis
        4. Areas of Agreement / Disagreement
        5. Limitations and Uncertainty
        6. Sources

        Every important factual claim should be traceable to a source URL.
        Do not invent citations, statistics, dates, or quotes.
        """,
        expected_output=(
            "A polished, evidence-based research report (under 700 words) "
            "with source URLs and clear caveats."
        ),
        agent=writer,
        context=[research_task, checking_task],
    )

    crew = Crew(
        agents=[planner, researcher, checker, writer],
        tasks=[planning_task, research_task, checking_task, writing_task],
        process=Process.sequential,
        max_rpm=MAX_RPM,
        verbose=True,
    )

    return crew.kickoff()
