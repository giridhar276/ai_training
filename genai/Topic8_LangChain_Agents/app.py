import streamlit as st
#from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import AgentType
#from langchain.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool

def main():
  st.set_page_config(page_title="AI Assistant ", page_icon=":robot:")
  st.title("AI Assistant :wave:")

  with st.sidebar:
    st.sidebar.title("Your File :page_with_curl:")
    input_csv=st.file_uploader("Upload a CSV file here", type=["csv"])

  if input_csv:
    user_question=st.text_input("Ask a question on your data")
    if user_question is not None:
      agent=create_csv_agent(OpenAI(temperature=0),
                            input_csv,
                            allow_dangerous_code=True,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=True)
      result=agent.run(user_question)
      st.write(result)



if __name__ == "__main__":
  main()
