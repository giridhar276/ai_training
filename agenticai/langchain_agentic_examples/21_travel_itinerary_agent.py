"""
Example 4: Travel Itinerary Planning Agent

Use case:
A user wants a 2-day Bangalore trip plan.

Agentic AI idea:
The agent decides whether to:
1. Check attractions
2. Check food options
3. Create day-wise itinerary

This is a mock planning demo.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")


def find_attractions(city: str) -> str:
    return """
Popular Bangalore Attractions:
- Lalbagh Botanical Garden
- Cubbon Park
- Bangalore Palace
- Visvesvaraya Museum
- Commercial Street
- UB City
"""


def find_food_options(preference: str) -> str:
    return """
Food Options:
- Breakfast: MTR or Vidyarthi Bhavan
- Lunch: Nagarjuna Andhra meals
- Snacks: CTR dosa
- Dinner: Church Street cafes
"""


def create_itinerary(details: str) -> str:
    return """
2-Day Bangalore Itinerary:
Day 1:
Morning: Lalbagh
Afternoon: Bangalore Palace
Evening: Church Street and dinner

Day 2:
Morning: Cubbon Park
Afternoon: Visvesvaraya Museum
Evening: Commercial Street shopping
"""


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

tools = [
    Tool(
        name="FindAttractions",
        func=find_attractions,
        description="Use this to find tourist attractions in a city.",
    ),
    Tool(
        name="FindFoodOptions",
        func=find_food_options,
        description="Use this to find food options based on preference.",
    ),
    Tool(
        name="CreateItinerary",
        func=create_itinerary,
        description="Use this to create a day-wise travel itinerary.",
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
)

user_request = """
Create a simple 2-day Bangalore trip plan.
I like parks, museums, local food, and shopping.
Keep it practical and not too expensive.
"""

result = agent.invoke(user_request)
print("\nFINAL ANSWER:\n")
print(result["output"])
