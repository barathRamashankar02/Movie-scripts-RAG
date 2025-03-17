from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import streamlit as st

from PyPDF2 import PdfReader
import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from uuid import uuid4
from langchain.chat_models import init_chat_model
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

#embedding function
def get_embeddings():
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-ada-002")
    return embeddings

#calling vectore store
vector_store = Chroma(
    collection_name="movie-scripts",
    embedding_function=get_embeddings(),
    persist_directory="movie-scripts-vdb"
)

#invoking model
llm = init_chat_model(
    "llama3-8b-8192", 
    model_provider="groq")


def rag_retreiver(question):
    #retrieving and generating answer
    prompt = hub.pull("rlm/rag-prompt")
    question = question
    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = prompt.invoke({"question": question, "context": docs_content})
    answer = llm.invoke(prompt)
    return answer.content

chat_history = []

system_message = SystemMessage(content="Answer the following questions based on the context")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    human_message = HumanMessage(content=query)
    chat_history.append(human_message)
    ai_message = rag_retreiver(query)
    chat_history.append(AIMessage(content=ai_message))
    print(f"AI: {ai_message}")

print("------------Msg Hisotry------------")
for msg in chat_history:
    print(msg.content)
