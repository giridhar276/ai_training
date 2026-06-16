import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

def read_csv_columns(file_name: str) -> str:
    """Read a CSV file and return its columns."""
    df = pd.read_csv(file_name)
    return f"Columns: {list(df.columns)} | Rows: {len(df)}"

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

agent = create_agent(model, tools=[read_csv_columns])

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Check students.csv and tell me the columns and number of rows."}
    ]
})

print(result)
