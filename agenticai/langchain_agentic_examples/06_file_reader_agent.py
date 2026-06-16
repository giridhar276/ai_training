

# agent uses external data

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

def read_text_file(file_path: str) -> str:
    """Read text from a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def short_summary(text: str) -> str:
    """Return a short summary of text."""
    return text[:200] + "..."

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

agent = create_agent(model, tools=[read_text_file, short_summary])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Read sample.txt and give me a short summary in a line."}
    ]
})

print(result["messages"][-1].content)
