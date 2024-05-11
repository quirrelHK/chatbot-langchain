import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries."),
        ("user","Question:{question}"),
    ]
)

# streamlit framework
st.title("Chatbot tinyllama")
input_text = st.text_input("How may I help you?")

# llama LLM
llm = Ollama(model="tinyllama")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
