import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your Gemini API key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyDaDvsul0_3mBWGJCnULJ1662nwrCATtEk"  

# Initialize the LangChain + Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Sample job postings dataset
data = {
    "title": ["Python Developer", "Data Scientist Intern", "Remote Web Developer"],
    "location": ["Mumbai", "Bangalore", "Remote"],
    "skills": ["Python, Django", "Python, ML, Data Analysis", "HTML, CSS, JavaScript"],
    "experience_required": ["2 years", "Internship", "1 year"]
}
jobs_df = pd.DataFrame(data)

# Streamlit UI

st.title("AI-Powered Job Recommendation System")

# User Inputs
skills = st.text_input("Enter your skills (comma separated):")
experience = st.text_input("Enter your experience (e.g., Internship, 2 years):")
location = st.text_input("Preferred job location:")
job_type = st.selectbox("Job Type:", ["Full-time", "Internship", "Remote"])

if st.button("Find Jobs"):
    # Prepare job data for AI prompt
    job_data_text = "\n".join([
        f"{row.title} in {row.location} - Skills: {row.skills} - Experience Required: {row.experience_required}"
        for _, row in jobs_df.iterrows()
    ])

    # Prepare prompt for Gemini
    prompt = f"""
    You are an expert job recommendation assistant.
    Here are some job postings:
    {job_data_text}

    The user has these skills: {skills}
    Experience: {experience}
    Preferred location: {location}
    Job type preference: {job_type}
    
    Recommend the 3 best matching jobs and explain briefly why each is a good fit.
    """

    # Get AI recommendations
    with st.spinner("Generating recommendations..."):
        recommendations = llm.predict(prompt)

    st.subheader("Recommended Jobs")
    st.write(recommendations)
