# agent1 : generates content 
# agent2 : summarizes content for presentation

# ageng colloboration


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

topic = "Benefits of agentic AI in customer support"

research_output = research_agent.invoke(
    f"Explain {topic} in 6 simple bullet points."
).content

summary_output = summary_agent.invoke(
    f"Summarize this in 4 simple lines for students:\n\n{research_output}"
).content

print("Research Agent Output:\n")
print(research_output)
print("\nSummary Agent Output:\n")
print(summary_output)




# I wanta  plan to travel to Hyderabad to Delhi where the travel price is less than 10000rs 


agent1:  find all the flghts between hyderabad to delhi 
agent2: get all the price details 
agent3: find the cheapest flight 
agent4: checking weather information 
agent5: