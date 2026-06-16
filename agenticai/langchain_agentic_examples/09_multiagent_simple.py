# agent collaboration

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

research_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

summary_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

topic = "LangChain agents"

research = research_agent.invoke(
    f"Give 5 simple points about {topic} for students."
).content

summary = summary_agent.invoke(
    f"Summarize this in 3 short lines:\n\n{research}"
).content

print("Research Agent Output:\n", research)
print("\nSummary Agent Output:\n", summary)
