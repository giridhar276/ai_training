

'''
Resume Screening Team

Skills agent checks technical fit
Culture agent checks collaboration fit
Recruiter agent gives final summary

Key concept:
Parallel review and final decision
'''

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

skills_agent = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
culture_agent = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
recruiter_agent = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)

candidate = '''
Candidate: Priya
Experience: 5 years
Skills: Python, LangChain, APIs, FastAPI, Pandas
Behavior: Mentored juniors, handled client demos, worked under tight deadlines
'''

skills_review = skills_agent.invoke(
    f"Review this candidate only for technical skills in 4 short points:\n{candidate}"
).content

culture_review = culture_agent.invoke(
    f"Review this candidate only for collaboration and attitude in 4 short points:\n{candidate}"
).content

final_summary = recruiter_agent.invoke(
    f"Combine these two reviews into a final recruiter recommendation.\nSkills Review:\n{skills_review}\n\nCulture Review:\n{culture_review}"
).content

print("SKILLS REVIEW:\n", skills_review)
print("\nCULTURE REVIEW:\n", culture_review)
print("\nFINAL SUMMARY:\n", final_summary)
