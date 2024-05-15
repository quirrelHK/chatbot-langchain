import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.environ["GROQ_API_KEY"]

def doc_vector_embeddings():
    
    if "vector_db_doc" not in st.session_state:
        st.session_state.loader = PyPDFDirectoryLoader("../huggingface/us_census")
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting
        st.session_state.vector_db_doc=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector embeddings
        
        st.session_state.vector_db.merge_from(st.session_state.vector_db_doc)
        
        
        
    

if "vector_db" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="tinyllama")
    st.session_state.loader = WebBaseLoader(web_path="https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vector_db = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
st.title("Langchain with Groq")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Please response to the user questions.
    Answer the questions based on the provided context only.
    Provide the most accurate response basedon the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)


prompt = st.text_input("Input your query here")

if st.button("Create document embeddings"):
    doc_vector_embeddings()
    st.write("Vector db of documents created")

if prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    end_time = time.process_time()
    print("Response time: ", end_time-start_time)
    
    st.write(response["answer"])
    
    with st.expander("Document Similarity Search"):
        
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")