import os
import json
import pandas as pd
import traceback
import PyPDF2

# Function to read files (PDF or TXT)
def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception("Error reading the PDF file")
        
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    
    else:
        raise Exception("Unsupported file format. Only PDF and text files are supported.")

# Function to convert quiz JSON to table format
def get_table_data(quiz_str):
    try:
        if isinstance(quiz_str, str):
            quiz_dict = json.loads(quiz_str)
        else:
            quiz_dict = quiz_str

        quiz_table_data = []

        # Extract each question's details
        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " || ".join(
                [f"{option} -> {option_value}" for option, option_value in value["options"].items()]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
        
        return quiz_table_data
    
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return None
