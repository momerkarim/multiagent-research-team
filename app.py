import streamlit as st
from crew import run_research

st.set_page_config(page_title="Multi-Agent Researcher", page_icon="🔎", layout="wide")

st.title("🔎 Multi-Agent Research Team")
st.caption("CrewAI + Groq GPT-OSS 120B + Web Research Tools")

with st.sidebar:
    st.header("Research Settings")
    st.write("The team uses four specialized agents:")
    st.markdown("""
    1. **Research Planner** – breaks the question into research angles
    2. **Web Researcher** – gathers evidence from the web
    3. **Fact Checker** – cross-checks important claims
    4. **Research Writer** – produces the final report
    """)
    st.info("Set GROQ_API_KEY and SERPER_API_KEY in Streamlit Secrets before running.")

query = st.text_area(
    "What would you like the research team to investigate?",
    placeholder="Example: What are the major trends in enterprise generative AI adoption in 2026?",
    height=140,
)

if st.button("🚀 Start Research", type="primary", disabled=not query.strip()):
    with st.spinner("The research team is working..."):
        try:
            result = run_research(query.strip())
            st.success("Research completed")
            st.markdown("## Research Report")
            st.markdown(str(result))
        except Exception as e:
            st.error(f"Research failed: {e}")
            st.exception(e)
