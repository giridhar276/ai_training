



import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

# ----------------------------------------
# Shared LLM model
# ----------------------------------------
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

# ----------------------------------------
# Agent 1: Delhi price estimator
# ----------------------------------------
delhi_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a travel cost estimator for Delhi trips. "
        "Estimate a low-budget 3-day trip cost from Hyderabad to Delhi. "
        "Include transport, stay, food, and local travel. "
        "Return the answer in this format:\n"
        "City: Delhi\n"
        "Transport: ₹...\n"
        "Stay: ₹...\n"
        "Food: ₹...\n"
        "Local Travel: ₹...\n"
        "Total: ₹...\n"
        "Reason: ..."
    )
)

# ----------------------------------------
# Agent 2: Kochi price estimator
# ----------------------------------------
kochi_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a travel cost estimator for Kochi trips. "
        "Estimate a low-budget 3-day trip cost from Hyderabad to Kochi. "
        "Include transport, stay, food, and local travel. "
        "Return the answer in this format:\n"
        "City: Kochi\n"
        "Transport: ₹...\n"
        "Stay: ₹...\n"
        "Food: ₹...\n"
        "Local Travel: ₹...\n"
        "Total: ₹...\n"
        "Reason: ..."
    )
)

# ----------------------------------------
# Agent 3: Chennai price estimator
# ----------------------------------------
chennai_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a travel cost estimator for Chennai trips. "
        "Estimate a low-budget 3-day trip cost from Hyderabad to Chennai. "
        "Include transport, stay, food, and local travel. "
        "Return the answer in this format:\n"
        "City: Chennai\n"
        "Transport: ₹...\n"
        "Stay: ₹...\n"
        "Food: ₹...\n"
        "Local Travel: ₹...\n"
        "Total: ₹...\n"
        "Reason: ..."
    )
)

# ----------------------------------------
# Final agent: compare and choose cheapest
# ----------------------------------------
final_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a final travel comparison agent. "
        "You will receive travel budget outputs for Delhi, Kochi, and Chennai. "
        "Compare all total costs and choose the cheapest city. "
        "Return the answer in this format:\n"
        "Delhi Total: ₹...\n"
        "Kochi Total: ₹...\n"
        "Chennai Total: ₹...\n"
        "Cheapest City: ...\n"
        "Why: ...\n"
        "Suggested Travel Plan: ..."
    )
)

# ----------------------------------------
# Common user request
# ----------------------------------------
trip_request = (
    "Estimate a 3-day budget trip from Hyderabad. "
    "Assume economical transport, budget stay, simple food, and low-cost local travel."
)

# ----------------------------------------
# Run all 3 city agents independently
# ----------------------------------------
delhi_result = delhi_agent.invoke({
    "messages": [{"role": "user", "content": trip_request}]
})

kochi_result = kochi_agent.invoke({
    "messages": [{"role": "user", "content": trip_request}]
})

chennai_result = chennai_agent.invoke({
    "messages": [{"role": "user", "content": trip_request}]
})

delhi_text = delhi_result["messages"][-1].content
kochi_text = kochi_result["messages"][-1].content
chennai_text = chennai_result["messages"][-1].content

# ----------------------------------------
# Pass all three outputs to final agent
# ----------------------------------------
comparison_prompt = f"""
Compare these 3 travel plans and choose the cheapest.

Delhi Agent Output:
{delhi_text}

Kochi Agent Output:
{kochi_text}

Chennai Agent Output:
{chennai_text}
"""

final_result = final_agent.invoke({
    "messages": [{"role": "user", "content": comparison_prompt}]
})

# ----------------------------------------
# Print outputs
# ----------------------------------------
print("=== DELHI AGENT OUTPUT ===")
print(delhi_text)

print("\n=== KOCHI AGENT OUTPUT ===")
print(kochi_text)

print("\n=== CHENNAI AGENT OUTPUT ===")
print(chennai_text)

print("\n=== FINAL AGENT OUTPUT ===")
print(final_result["messages"][-1].content)