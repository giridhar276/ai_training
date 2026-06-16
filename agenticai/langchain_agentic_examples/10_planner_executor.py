
# task decomposition
# planner + executor 
# planner creates steps
# executor explains/executes


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

planner = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

executor = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

goal = "Teach Agentic AI in a 15-minute demo"

plan = planner.invoke(
    f"Create 5 short steps to achieve this goal: {goal}"
).content

execution_help = executor.invoke(
    f"Based on this plan, suggest what I should say for each step:\n\n{plan}"
).content

print("Plan:\n", plan)
print("\nExecution Help:\n", execution_help)
