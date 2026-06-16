# decision making agent that can use tools to perform calculations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

agent = create_agent(model, tools=[add, multiply])

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is 12 plus 8, and then multiply the result by 2?"
        }
    ]
})

print(result["messages"][-1].content)



