

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# Load API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

# Shared model
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

# -----------------------------
# Independent Agent 1: Research Agent
# -----------------------------
research_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a research agent. "
        "Your job is to collect useful travel-related information "
        "such as places to visit, local transport, and timing suggestions. "
        "Return short, structured findings."
    )
)

@tool
def ask_research_agent(query: str) -> str:
    """Use this for travel research, sightseeing suggestions, and local planning."""
    result = research_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    return result["messages"][-1].content


# -----------------------------
# Independent Agent 2: Budget Agent
# -----------------------------
budget_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a budget planning agent. "
        "Your job is to estimate costs, split expenses into categories, "
        "and keep the plan within the given budget. "
        "Return clear cost breakdowns."
    )
)

@tool
def ask_budget_agent(query: str) -> str:
    """Use this for cost estimation, expense splitting, and budget optimization."""
    result = budget_agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    return result["messages"][-1].content


# -----------------------------
# Main Supervisor Agent
# -----------------------------
main_agent = create_agent(
    model=model,
    tools=[ask_research_agent, ask_budget_agent],
    system_prompt=(
        "You are a supervisor agent. "
        "Delegate travel research tasks to ask_research_agent. "
        "Delegate budget and cost tasks to ask_budget_agent. "
        "Combine the final answer into one neat response."
    )
)

# -----------------------------
# Example Run
# -----------------------------
response = main_agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Create a 2-day plan either for Delhi for cochin or chennai  plan under 10000 INR. "
                "Suggest places to visit and also provide a cost breakdown."

            )
        }
    ]
})

print(response["messages"][-1].content)