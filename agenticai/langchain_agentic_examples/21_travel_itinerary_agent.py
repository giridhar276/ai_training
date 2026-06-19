"""
Travel Itinerary Planning Agent using LangChain

Use case:
The agent plans a 2-day trip by checking weather, attractions,
hotel suggestions, and daily itinerary.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please add it in your .env file.")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)


@tool
def check_weather(city: str, travel_date: str) -> str:
    """Use this tool to check sample weather for a city and date."""
    return f"Weather in {city} on {travel_date}: 28°C, partly cloudy, light rain expected in evening."


@tool
def find_top_attractions(city: str) -> str:
    """Use this tool to find top attractions in a city."""
    attractions = {
        "bangalore": "Lalbagh Botanical Garden, Cubbon Park, Bangalore Palace, Church Street, UB City",
        "hyderabad": "Charminar, Golconda Fort, Hussain Sagar, Ramoji Film City, Salar Jung Museum",
        "mysore": "Mysore Palace, Chamundi Hills, Brindavan Gardens, St. Philomena's Church"
    }
    return attractions.get(city.lower(), "Popular attractions not found in sample data.")


@tool
def suggest_hotels(city: str, budget: str) -> str:
    """Use this tool to suggest hotels based on city and budget."""
    return f"Hotel suggestions in {city} for {budget} budget: Comfort Stay, City Inn, Grand Residency."


@tool
def estimate_trip_cost(city: str, days: int) -> str:
    """Use this tool to estimate trip cost."""
    estimated_cost = days * 3500
    return f"Estimated trip cost for {days} days in {city}: ₹{estimated_cost}, excluding travel tickets."


tools = [check_weather, find_top_attractions, suggest_hotels, estimate_trip_cost]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a travel itinerary planning agent.
Your job:
- Understand destination, number of days, budget, and travel date.
- Check weather.
- Find attractions.
- Suggest hotels.
- Estimate cost.
- Create a simple day-wise itinerary.
"""
)

question = "Plan a 2-day Bangalore trip for 25 June 2026. My budget is medium. Include places, hotel suggestion, weather and estimated cost."

print("\nCustomer Request:")
print(question)
response = agent.invoke({"messages": [{"role": "user", "content": question}]})
print("\nAgent Response:")
print(response["messages"][-1].content)
