"""
Example 5: Employee Expense Approval Agent

Use case:
An employee submits travel expenses for approval.

Agentic AI idea:
The agent decides whether to:
1. Check company policy
2. Validate submitted bills
3. Approve, reject, or request clarification

This is a mock enterprise workflow demo.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Add it inside your .env file.")


def check_expense_policy(expense_text: str) -> str:
    return """
Company Expense Policy:
- Local taxi limit per day: ₹2,000
- Food limit per day: ₹1,500
- Hotel limit per night: ₹5,000
- Original bills are required for amounts above ₹1,000
"""


def validate_bills(expense_text: str) -> str:
    return """
Bill Validation:
- Taxi bill: ₹1,800, bill attached, valid
- Food bill: ₹1,200, bill attached, valid
- Hotel bill: ₹6,200, bill attached, exceeds policy limit
"""


def approval_decision(expense_text: str) -> str:
    return """
Approval Decision:
Partially approved.
Approved:
- Taxi: ₹1,800
- Food: ₹1,200
- Hotel: ₹5,000 as per policy limit
Not approved:
- Extra hotel amount: ₹1,200
Reason: Hotel expense exceeds allowed limit.
"""


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

tools = [
    Tool(
        name="CheckExpensePolicy",
        func=check_expense_policy,
        description="Use this to check company expense policy.",
    ),
    Tool(
        name="ValidateBills",
        func=validate_bills,
        description="Use this to validate submitted bills.",
    ),
    Tool(
        name="ApprovalDecision",
        func=approval_decision,
        description="Use this to approve, reject, or partially approve expenses.",
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
Employee submitted travel expense:
Taxi: ₹1800 with bill
Food: ₹1200 with bill
Hotel: ₹6200 with bill
Please check policy and decide approval.
"""

result = agent.invoke(user_request)
print("\nFINAL ANSWER:\n")
print(result["output"])
