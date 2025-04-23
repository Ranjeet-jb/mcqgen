import os
import re
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
import streamlit as st
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

# Load response format JSON
with open(r"C:\Users\Ranjeet Kumar\Desktop\mcqgen\Response.json", "r") as file:
    RESPONSE_JSON = json.load(file)

# Streamlit App Title
st.title("MCQs Creator Application with LangChain + Gemini 🤖")

# Form to collect user input
with st.form("user_inputs"):

    uploaded_file = st.file_uploader("Upload a PDF or text file")
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="simple")
    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Generating MCQs using Gemini..."):
            try:
                # Step 1: Extract text
                text = read_file(uploaded_file)

                # Step 2: Generate using Gemini via RunnableSequence
                response = generate_evaluate_chain.invoke({
                    "text": text,
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON)
                })

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error while generating MCQs.")

            else:
                if isinstance(response, dict):
                    quiz = response.get("quiz", None)

                            # Try parsing quiz if it's a JSON string
                    if isinstance(quiz, str):
                        try:
                                # Extract JSON-like content using regex
                            match = re.search(r"\{(?:.|\n)*\}", quiz)
                            if match:
                                clean_json = match.group(0)
                                quiz = json.loads(clean_json)  # quiz is now a dict ✅
                            else:
                                st.error("⚠️ JSON format not detected in quiz output.")
                                quiz = None
                        except json.JSONDecodeError:
                            st.error("⚠️ Failed to decode quiz JSON.")
                            st.code(quiz, language="json")
                            quiz = None

                    if isinstance(quiz, dict):  # 💡 only call get_table_data if it's a dict
                        table_data = get_table_data(quiz)  # pass dict, NOT again json.loads
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)

                            review = response.get("review", "")
                            st.subheader("Review")
                            st.text_area(label="Review", value=review, height=150)
                        else:
                            st.error("Error in generating table data.")
                    else:
                        st.error("⚠️ No quiz found in the response.")
                else:
                    st.write("⚠️ Unexpected response format.")