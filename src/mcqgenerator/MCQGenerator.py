import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',
    google_api_key=api_key,
    temperature=0.0
)

# Template for quiz generation
template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. 
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

# Define the prompt template
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template
)

# Initialize the LLM chain
quiz_chain = LLMChain(
    llm=llm,
    prompt=quiz_generation_prompt,
    output_key="quiz",
    verbose=True
)

# Template for quiz evaluation
template2 = """
You are an expert English grammarian and writer. Given a multiple choice Quiz for {subject} students.
You need to evaluate the complexity of the questions and give a complete analysis of the quiz. Only use at max 50 words for complexity.
If the Quiz is not at par with the cognitive and analytical abilities of the students, update the quiz questions which need to be changed and change the tone such that it perfectly fits the student's abilities.
Quiz_MCQs:
{quiz}

Check from an Expert English Writer of the above quiz:
"""

# Define the evaluation prompt template
quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2
)

# Initialize the review chain
review_chain = LLMChain(
    llm=llm,
    prompt=quiz_evaluation_prompt,
    output_key="review",
    verbose=True
)

# Combine both chains (generation + review)
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True
)