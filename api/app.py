import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

app = FastAPI(
    title="Langchain Srever",
    version="1.0",
    description="Langchain API",
)

add_routes(
    app,
    Ollama(model="tinyllama"),
    path="/tinyllama"
)

llm_llama = Ollama(model="tinyllama")
llm_dolphin = Ollama(model="tinydolphin")

prompt1 = ChatPromptTemplate.from_template("Write me an essay on the topic {topic} with 100 words.")
prompt2 = ChatPromptTemplate.from_template("Write me a poem on the topic {topic} with 100 words.")

add_routes(
    app,
    prompt1|llm_llama,
    path="/essay"
)


add_routes(
    app,
    prompt2|llm_dolphin,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)