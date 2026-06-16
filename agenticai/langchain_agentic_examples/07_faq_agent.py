import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

def search_faq(question: str) -> str:
    """Search FAQ answers."""
    faq = {
        "what is python": "Python is a programming language.",
        "what is langchain": "LangChain helps build LAG applications.",
        "what is agentic ai": "Agentic AI uses reasoning plus tools and actions."
    }
    return faq.get(question.lower(), "No FAQ answer found.")

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

agent = create_agent(model, tools=[search_faq])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is Agentic AI?"}
    ]
})

print(result["messages"][-1].content)
