"""
Banking Customer Support Agent using LangChain

Use cases:
1. Check account balance
2. Show recent transactions
3. Block debit card
4. Check loan EMI
5. Create support ticket
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
def check_account_balance(account_number: str) -> str:
    """Use this tool when customer asks for account balance."""
    return f"Account number {account_number} has current balance of ₹52,300."


@tool
def get_recent_transactions(account_number: str) -> str:
    """Use this tool when customer asks for recent transactions."""
    return f"""
Recent transactions for account {account_number}:

1. Salary Credit: ₹85,000
2. ATM Withdrawal: ₹5,000
3. Amazon Purchase: ₹2,400
4. Electricity Bill Payment: ₹3,200
5. UPI Transfer: ₹1,500
"""


@tool
def block_debit_card(card_number: str) -> str:
    """Use this tool when customer wants to block debit card."""
    return f"Debit card ending with {card_number[-4:]} has been blocked successfully."


@tool
def check_loan_emi(loan_account_number: str) -> str:
    """Use this tool when customer asks about loan EMI details."""
    return f"""
Loan account {loan_account_number} EMI details:

Monthly EMI: ₹18,500
Due Date: 5th of every month
Outstanding Loan Amount: ₹6,75,000
Loan Status: Active
"""


@tool
def create_support_ticket(issue_description: str) -> str:
    """Use this tool when customer reports an issue or complaint."""
    return f"""
Support ticket created successfully.

Ticket ID: BNK20260091
Issue: {issue_description}
Status: Open
Expected Resolution Time: 24 to 48 hours
"""


tools = [check_account_balance, get_recent_transactions, block_debit_card, check_loan_emi, create_support_ticket]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are a helpful banking customer support agent.
Your job:
- Understand the customer request.
- Select the correct banking tool.
- Use the tool result to answer clearly.
- Keep the answer short and professional.
- Do not invent account data.
- If required information is missing, ask the customer for it.
"""
)

questions = [
    "My account number is 1234567890. What is my balance?",
    "Show recent transactions for account number 1234567890.",
    "Please block my debit card number 4567891234567890.",
    "Check EMI for loan account LN908877.",
    "I was charged twice for my electricity bill payment. Please raise a complaint."
]

for question in questions:
    print("\nCustomer Question:")
    print(question)
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("\nAgent Response:")
    print(response["messages"][-1].content)
    print("-" * 60)
