import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries."),
        ("user","Question:{question}"),
    ]
)

# streamlit framework

st.title("OpenAI chatbot")
input_text = st.text_input("How may I help you?")

# OpenAI LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
