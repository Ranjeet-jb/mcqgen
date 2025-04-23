from setuptools import find_packages, setup

setup(
    name="mcqgenerator",
    version="0.0.1",
    author="ranjeet jb",
    author_email="ranjeetjaryalbhattbhatt@gmail.com",
    install_requires=[
        "langchain",
        "langchain-google-genai",
        "google-generativeai",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "pandas"  # You used it in the code for DataFrame and CSV export
    ],
    packages=find_packages()
)