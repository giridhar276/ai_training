"""
Employee Expense Approval Agent using LangChain

Use case:
The agent checks an employee reimbursement request,
validates policy, checks amount limits, and approves or rejects.
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
def check_expense_policy(expense_type: str) -> str:
    """Use this tool to check whether an expense type is allowed."""
    allowed = ["travel", "hotel", "food", "client meeting", "training"]
    if expense_type.lower() in allowed:
        return f"{expense_type} expense is allowed as per company policy."
    return f"{expense_type} expense is not clearly allowed as per company policy."


@tool
def check_amount_limit(expense_type: str, amount: int) -> str:
    """Use this tool to check whether the expense amount is within limit."""
    limits = {
        "travel": 15000,
        "hotel": 8000,
        "food": 2000,
        "client meeting": 5000,
        "training": 20000
    }
    limit = limits.get(expense_type.lower(), 0)
    if limit == 0:
        return "No amount limit found because expense type is unknown."
    if amount <= limit:
        return f"Amount ₹{amount} is within the allowed limit of ₹{limit}."
    return f"Amount ₹{amount} exceeds the allowed limit of ₹{limit}."


@tool
def check_receipt_status(receipt_attached: str) -> str:
    """Use this tool to check whether receipt is attached."""
    if receipt_attached.lower() in ["yes", "true", "attached"]:
        return "Receipt is attached."
    return "Receipt is missing."


@tool
def create_approval_record(employee_name: str, decision: str) -> str:
    """Use this tool to create final approval or rejection record."""
    return f"Expense decision recorded for {employee_name}: {decision}. Reference ID: EXP20260456"


tools = [check_expense_policy, check_amount_limit, check_receipt_status, create_approval_record]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an employee expense approval agent.
Your job:
- Check expense type policy.
- Check amount limit.
- Check receipt status.
- Approve if policy is allowed, amount is within limit, and receipt is attached.
- Reject or ask for more information if something is missing.
- Keep answer professional.
"""
)

expense_request = """
Employee Name: Raj Kumar
Expense Type: travel
Amount: 12500
Receipt Attached: yes
Purpose: Client visit from Hyderabad to Bangalore
"""

question = f"Review this expense request and approve or reject:\n{expense_request}"

print("\nManager Request:")
print(question)
response = agent.invoke({"messages": [{"role": "user", "content": question}]})
print("\nAgent Response:")
print(response["messages"][-1].content)
