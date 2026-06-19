"""
Example 1: Ticket Booking Agent

Use case:
A user wants to book a ticket from Hyderabad to Bangalore.

Agentic AI idea:
The LLM does not only answer. It decides which tool to use:
1. Search available tickets
2. Check user budget
3. Create a booking summary

This is a safe classroom demo. It does not book a real ticket.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")


def search_tickets(route: str) -> str:
    """Search mock tickets for a route."""
    return """
Available tickets for Hyderabad to Bangalore:
1. Flight: IndiFly 6E-204, 07:30 AM, 1h 20m, ₹4,800
2. Train: Kacheguda Express, 08:00 PM, 11h 30m, ₹1,250
3. Bus: Orange Travels, 10:30 PM, 9h 15m, ₹1,100
"""


def check_budget(user_details: str) -> str:
    """Check budget preference from user details."""
    return """
User budget preference detected:
- Budget: under ₹2,000
- Preferred mode: comfortable and overnight travel
- Best option: Bus or Train
"""


def create_booking_summary(selection: str) -> str:
    """Create a safe booking summary."""
    return """
Booking Summary:
Route: Hyderabad to Bangalore
Recommended ticket: Orange Travels Bus
Departure: 10:30 PM
Duration: 9h 15m
Price: ₹1,100
Status: Draft booking only. Payment and final confirmation are not done.
"""


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

tools = [
    Tool(
        name="SearchTickets",
        func=search_tickets,
        description="Use this to search available tickets between two cities.",
    ),
    Tool(
        name="CheckBudget",
        func=check_budget,
        description="Use this to understand budget and travel preference.",
    ),
    Tool(
        name="CreateBookingSummary",
        func=create_booking_summary,
        description="Use this to create final draft booking summary.",
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
I want to travel from Hyderabad to Bangalore tomorrow.
My budget is below 2000 rupees.
I prefer comfortable overnight travel.
Find the best option and create a booking summary.
"""

result = agent.invoke(user_request)
print("\nFINAL ANSWER:\n")
print(result["output"])
