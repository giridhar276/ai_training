"""
Example 2: Banking Customer Support Agent

Use case:
A banking customer says their debit card transaction failed but money was deducted.

Agentic AI idea:
The agent decides whether to:
1. Check account transaction status
2. Check dispute policy
3. Create a support ticket

This is a safe mock demo. No real bank data is accessed.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")


def check_transaction_status(transaction_id: str) -> str:
    return """
Transaction Status:
Transaction ID: TXN90871
Amount: ₹3,499
Status: Failed at merchant side
Bank debit: Successful
Reversal expected: 2 to 5 working days
"""


def check_dispute_policy(issue: str) -> str:
    return """
Banking Policy:
If amount is debited but merchant transaction failed, customer can raise a dispute.
Expected resolution time: 5 to 7 working days.
Required details: transaction ID, amount, date, merchant name.
"""


def create_support_ticket(ticket_details: str) -> str:
    return """
Support Ticket Created:
Ticket ID: CASE-2026-4471
Category: Failed transaction with debit
Priority: Medium
Next step: Customer will receive SMS/email update within 24 hours.
"""


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

tools = [
    Tool(
        name="CheckTransactionStatus",
        func=check_transaction_status,
        description="Use this to check mock banking transaction status.",
    ),
    Tool(
        name="CheckDisputePolicy",
        func=check_dispute_policy,
        description="Use this to check bank dispute policy.",
    ),
    Tool(
        name="CreateSupportTicket",
        func=create_support_ticket,
        description="Use this to create a support ticket for the customer.",
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
Customer says: My debit card transaction TXN90871 failed at the shop,
but ₹3499 was deducted from my bank account. Please help me.
"""

result = agent.invoke(user_request)
print("\nFINAL ANSWER:\n")
print(result["output"])
