import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

planner_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

executor_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

goal = "Build a simple FAQ bot using LangChain"

plan = planner_agent.invoke(
    f"Create 5 beginner-friendly steps for this goal: {goal}"
).content

execution = executor_agent.invoke(
    f"Using this plan, explain what to do in each step in very simple words:\n\n{plan}"
).content

print("Planner Agent Output:\n")
print(plan)
print("\nExecutor Agent Output:\n")
print(execution)
