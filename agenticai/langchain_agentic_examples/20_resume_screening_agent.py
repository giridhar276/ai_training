"""
Example 3: Resume Screening Agent

Use case:
A recruiter wants to screen a candidate for a Python + LangChain role.

Agentic AI idea:
The agent decides whether to:
1. Check technical fit
2. Check experience fit
3. Give final recruiter recommendation

This is a mock HR screening demo.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")


def check_technical_fit(candidate_profile: str) -> str:
    return """
Technical Fit:
- Python: Strong
- Pandas: Good
- LangChain: Basic to intermediate
- APIs: Good
- Missing: LangGraph production experience
Technical score: 7.5/10
"""


def check_experience_fit(candidate_profile: str) -> str:
    return """
Experience Fit:
- Total experience: 5 years
- Relevant AI/ML experience: 2 years
- Client communication: Good
- Training/mentoring: Yes
Experience score: 8/10
"""


def recruiter_recommendation(candidate_profile: str) -> str:
    return """
Recruiter Recommendation:
Shortlist the candidate for technical interview.
Focus interview questions on LangChain tools, agents, RAG, APIs, and deployment.
"""


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

tools = [
    Tool(
        name="CheckTechnicalFit",
        func=check_technical_fit,
        description="Use this to evaluate technical skills of a candidate.",
    ),
    Tool(
        name="CheckExperienceFit",
        func=check_experience_fit,
        description="Use this to evaluate experience and project fit.",
    ),
    Tool(
        name="RecruiterRecommendation",
        func=recruiter_recommendation,
        description="Use this to give final recruiter recommendation.",
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
)

candidate = """
Candidate: Priya
Experience: 5 years
Skills: Python, Pandas, FastAPI, LangChain basics, OpenAI API
Projects: Built chatbot, document summarizer, and customer support automation
Behavior: Mentored juniors and handled client demos
"""

result = agent.invoke("Screen this candidate for a Python LangChain developer role:\n" + candidate)
print("\nFINAL ANSWER:\n")
print(result["output"])
