"""
Resume Screening Agent using LangChain

Use case:
The agent screens a candidate profile, checks technical fit,
checks culture fit, and gives a hiring recommendation.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please add it in your .env file.")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)


@tool
def check_python_skills(candidate_profile: str) -> str:
    """Use this tool to evaluate Python and data skills from a candidate profile."""
    profile = candidate_profile.lower()
    score = 0
    if "python" in profile:
        score += 30
    if "pandas" in profile:
        score += 25
    if "machine learning" in profile or "ml" in profile:
        score += 20
    if "langchain" in profile:
        score += 25
    return f"Technical skill score: {score}/100 based on Python, Pandas, ML, and LangChain keywords."


@tool
def check_experience_level(candidate_profile: str) -> str:
    """Use this tool to evaluate candidate experience level."""
    profile = candidate_profile.lower()
    if "8 years" in profile or "10 years" in profile:
        return "Experience level: Senior candidate"
    if "5 years" in profile or "6 years" in profile:
        return "Experience level: Mid-level candidate"
    if "1 year" in profile or "2 years" in profile:
        return "Experience level: Junior candidate"
    return "Experience level: Not clearly mentioned"


@tool
def check_culture_fit(candidate_profile: str) -> str:
    """Use this tool to evaluate teamwork, mentoring, and communication fit."""
    profile = candidate_profile.lower()
    points = []
    if "mentored" in profile:
        points.append("Mentoring experience found")
    if "client" in profile:
        points.append("Client communication experience found")
    if "team" in profile:
        points.append("Team collaboration found")
    if not points:
        points.append("Culture fit signals are weak")
    return "Culture fit: " + ", ".join(points)


@tool
def generate_hr_summary(candidate_name: str, recommendation: str) -> str:
    """Use this tool to create final HR screening summary."""
    return f"Final HR Summary for {candidate_name}: {recommendation}"


tools = [check_python_skills, check_experience_level, check_culture_fit, generate_hr_summary]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a resume screening agent.
Your job:
- Review candidate profile.
- Use technical, experience, and culture tools.
- Give a final recommendation: Shortlist, Hold, or Reject.
- Keep the explanation simple and practical.
"""
)

candidate = """
Candidate Name: Priya Sharma
Experience: 5 years
Skills: Python, Pandas, Machine Learning, LangChain, FastAPI
Behavior: Mentored juniors, worked with client teams, handled demos
Role Applied: AI Engineer
"""

question = f"Screen this candidate and give final recommendation:\n{candidate}"

print("\nRecruiter Request:")
print(question)
response = agent.invoke({"messages": [{"role": "user", "content": question}]})
print("\nAgent Response:")
print(response["messages"][-1].content)
