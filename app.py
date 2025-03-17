import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

# Embedding function
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

# Creating vector store
vector_store = Chroma(
    collection_name="movie-scripts",
    embedding_function=get_embeddings(),
    persist_directory="movie-scripts-vdb"
)

# Loading chat model
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

def rag_retreiver(question):
    prompt = hub.pull("rlm/rag-prompt")
    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = prompt.invoke({"question": question, "context": docs_content})
    answer = llm.invoke(prompt)
    return answer.content

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Single chat input with unique key
prompt = st.chat_input("Say something", key="chat_input")

if prompt:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if prompt.lower() == "exit":
        st.write("Chat ended.")
        st.session_state.messages = []  # Clear chat history
    else:
        # Get response from RAG model
        msg = rag_retreiver(prompt)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(f"RAG: {msg}")
