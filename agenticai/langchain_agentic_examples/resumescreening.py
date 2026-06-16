



'''
Resume Screening Team - Streamlit App

User uploads a resume
Skills agent checks technical fit
Culture agent checks collaboration fit
Recruiter agent gives final summary

Key concept:
Parallel review and final decision
'''

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# For PDF reading
from PyPDF2 import PdfReader

# For DOCX reading
from docx import Document

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

# ---------------------------------------------------
# Create agents
# ---------------------------------------------------
skills_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

culture_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

recruiter_agent = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=api_key
)

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def read_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_resume_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return read_txt(uploaded_file)
    elif file_name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return read_docx(uploaded_file)
    else:
        return None

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.set_page_config(page_title="Resume Screening Team", layout="wide")

st.title("Resume Screening Team")
st.write("Upload a resume and let multiple AI agents review it.")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["txt", "pdf", "docx"]
)

if uploaded_file is not None:
    resume_text = extract_resume_text(uploaded_file)

    if not resume_text:
        st.error("Could not read the uploaded file.")
    else:
        st.subheader("Extracted Resume Text")
        st.text_area("Resume Content", resume_text, height=250)

        if st.button("Analyze Resume"):
            with st.spinner("Agents are reviewing the resume..."):
                skills_review = skills_agent.invoke(
                    f"""
                    Review this candidate only for technical skills.
                    Give 4 short bullet points.

                    Resume:
                    {resume_text}
                    """
                ).content

                culture_review = culture_agent.invoke(
                    f"""
                    Review this candidate only for collaboration, attitude, communication,
                    and team behavior.
                    Give 4 short bullet points.

                    Resume:
                    {resume_text}
                    """
                ).content

                final_summary = recruiter_agent.invoke(
                    f"""
                    Combine these two reviews into a final recruiter recommendation.

                    Skills Review:
                    {skills_review}

                    Culture Review:
                    {culture_review}

                    Give:
                    1. Overall recommendation
                    2. Strengths
                    3. Concerns
                    4. Final hiring suggestion
                    """
                ).content

            st.success("Analysis completed")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Skills Review")
                st.write(skills_review)

            with col2:
                st.subheader("Culture Review")
                st.write(culture_review)

            st.subheader("Final Recruiter Summary")
            st.write(final_summary)