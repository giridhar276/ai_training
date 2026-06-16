import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

memory = ConversationBufferMemory(return_messages=True)

memory.chat_memory.add_user_message("My name is Rita.")
memory.chat_memory.add_ai_message("Nice to meet you, Rita.")
memory.chat_memory.add_user_message("I am teaching  AI today.")

history = memory.load_memory_variables({})

print("Memory contents:")
for msg in history["history"]:
    print(type(msg).__name__, ":", msg.content)

response = llm.invoke(history["history"] + [
    {"role": "user", "content": "What am I teaching today?"}
])

print("\nModel answer:")
print(response.content)
