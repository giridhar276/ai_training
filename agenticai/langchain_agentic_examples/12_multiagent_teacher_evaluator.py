import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

teacher_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

evaluator_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

question = teacher_agent.invoke(
    "Generate one simple question for students on Agentic AI."
).content

student_answer = "Agentic AI uses tools and decisions to complete tasks."

evaluation = evaluator_agent.invoke(
    f"Question: {question}\nStudent answer: {student_answer}\nEvaluate in 2 simple lines."
).content

print("Teacher Agent Question:\n")
print(question)
print("\nEvaluator Agent Result:\n")
print(evaluation)
