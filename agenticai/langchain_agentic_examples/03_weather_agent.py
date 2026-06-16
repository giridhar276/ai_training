import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from  langchain.tools import tool
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

def get_weather(city: str) -> str:
    """Return weather information for a city."""
    fake_data = {
        "hyderabad": "hot",
        "delhi": "rainy",
        "mumbai": "humid"
    }
    return fake_data.get(city.lower(), "pleasant")

def suggest_action(weather: str) -> str:
    """Suggest an action based on weather."""
    weather = weather.lower()
    if weather == "hot":
        return "Carry water and wear light clothes."
    elif weather == "rainy":
        return "Carry an umbrella."
    elif weather == "humid":
        return "Stay hydrated."
    return "Enjoy your day."

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

agent = create_agent(model, tools=[get_weather, suggest_action])

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I am going to Hyderabad. Check the weather and suggest what I should do."
        }
    ]
})

print(result["messages"][-1].content)
