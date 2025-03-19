import os
import gradio as gr
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain import hub


load_dotenv()

# Embeddings
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

# Create Vector Store
vector_store = Chroma(
    collection_name="movie-scripts",
    embedding_function=get_embeddings(),
    persist_directory="movie-scripts-vdb"
)

# Load Chat Model (Change provider if needed)
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

# RAG Retriever Function
def rag_retreiver(question: str) -> str:
    """
    Perform similarity search on the vector store, build a prompt
    with retrieved docs, then invoke the LLM for an answer.
    """
    prompt = hub.pull("rlm/rag-prompt")  # using prompt from langchain hub FYI can be changed
    

    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = prompt.invoke({
        "question": question,
        "context": docs_content
    })
    
    answer = llm.invoke(prompt)
    return answer.content

# Gradio Predict Function
def predict(message, history):
    if message.strip().lower() == "exit":
        return "Chat ended. (Type anything else to start a new conversation.)"
    
    response = rag_retreiver(message)
    return response

demo = gr.ChatInterface(
    fn=predict,
    title="Movie Scripts RAG No Memory or interactive chat",
    type="messages"
)

demo.launch(inbrowser=True)
