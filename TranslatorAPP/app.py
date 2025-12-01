from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Only set if the environment variables exist
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
input_language = "English"
output_language = "Tamil"
## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("user", "Translate the following text: {text}"),
    ]
)

## streamlit framework
st.title("Language Translator")
input_text = st.text_input("Enter text to translate")

# Ollama LLM model (free and local)
llm = ChatOllama(model="llama3.2", temperature=0)
Output_Parser=StrOutputParser()
chain=prompt|llm|Output_Parser

if input_text:
    st.write(chain.invoke({
        'input_language': input_language,
        'output_language': output_language,
        'text': input_text
    }))