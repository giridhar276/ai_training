"""
Ticket Booking Agent using LangChain

Use case:
Customer wants to book a ticket from Hyderabad to Bangalore.
The agent checks available buses/flights, compares options, and creates a booking.

Install:
pip install langchain langchain-openai langchain-core python-dotenv

.env file:
OPENAI_API_KEY=your_api_key_here
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
def search_travel_options(source: str, destination: str, travel_date: str) -> str:
    """Use this tool to search travel options between two cities for a date."""
    return f"""
Available travel options from {source} to {destination} on {travel_date}:

1. Flight AI-456
   Departure: 07:30 AM
   Arrival: 08:45 AM
   Price: ₹4,800

2. Flight 6E-221
   Departure: 11:00 AM
   Arrival: 12:15 PM
   Price: ₹5,200

3. Bus Orange Travels
   Departure: 09:30 PM
   Arrival: 07:00 AM next day
   Price: ₹1,450

4. Train Kacheguda Express
   Departure: 06:00 PM
   Arrival: 06:30 AM next day
   Price: ₹1,200
"""


@tool
def check_seat_availability(option_name: str) -> str:
    """Use this tool to check seat availability for a selected travel option."""
    availability = {
        "Flight AI-456": "12 seats available",
        "Flight 6E-221": "5 seats available",
        "Bus Orange Travels": "18 seats available",
        "Train Kacheguda Express": "WL 8 waiting list"
    }
    return availability.get(option_name, "Availability not found for this option.")


@tool
def book_ticket(passenger_name: str, option_name: str) -> str:
    """Use this tool to book a ticket for the passenger on the selected travel option."""
    return f"""
Ticket booked successfully.

Passenger: {passenger_name}
Travel Option: {option_name}
Booking ID: TRV20261091
Status: Confirmed
"""


@tool
def cancel_ticket(booking_id: str) -> str:
    """Use this tool when customer wants to cancel a ticket."""
    return f"Booking {booking_id} has been cancelled successfully. Refund will be processed in 3-5 working days."


tools = [search_travel_options, check_seat_availability, book_ticket, cancel_ticket]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a travel booking agent.
Your job:
- Understand source, destination, date, and passenger name.
- Search options if needed.
- Check availability before booking.
- Book only when the customer clearly asks to book.
- Keep the response clear and professional.
"""
)

questions = [
    "I want to travel from Hyderabad to Bangalore on 25 June 2026. Show me options.",
    "Check seat availability for Flight AI-456.",
    "Book Flight AI-456 for Giridhar Sripathi.",
    "Cancel my booking TRV20261091."
]

for question in questions:
    print("\nCustomer Question:")
    print(question)
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("\nAgent Response:")
    print(response["messages"][-1].content)
    print("-" * 60)
