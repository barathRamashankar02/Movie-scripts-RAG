from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from PyPDF2 import PdfReader
import os
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from uuid import uuid4
from langchain.chat_models import init_chat_model
from langchain import hub
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import gradio as gr

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
# llm = init_chat_model(
#     "llama3-8b-8192", 
#     model_provider="groq")
llm = init_chat_model(
    "gpt-3.5-turbo", 
    model_provider="openai")

#stage graph
graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(question: str):
    """Retrieve relevant documents based on the input query."""
    retrieved_docs = vector_store.similarity_search(question)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
k
# nodes
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

#edges and combining nodes

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "message_thread_1"}}


def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    query = history_langchain_format[-1].content
    state = graph.invoke({"messages": [query]}, config=config)
    response = state["messages"][-1].content
    return response

demo = gr.ChatInterface(
    predict,
    type="messages"
)

demo.launch()


############################ terminal check ########################################
# if query:
#     # Store user message
#     st.session_state.chat_history.append(HumanMessage(query))
#     st.session_state.messages.append({"role": "user", "content": query})
#     st.chat_message("user").write(query)

#     if query.lower() == "exit":
#         st.write("Chat ended.")
#         st.session_state.messages = []  
    
#     else:
#         state = graph.invoke({"messages": st.session_state.chat_history}, config=config)
#         response = state["messages"][-1].content
#         st.session_state.messages.append({"role": "AI", "content": response})
#         st.session_state.chat_history.append(AIMessage(query))
#         st.chat_message("AI").write(f"AI: {response}")
# while True:
#     query = input("You: ")  
#     if query:
#         if query.lower() == "exit":
#             break
#         state = graph.invoke({"messages": [query]}, config=config)
#         response = state["messages"][-1].content
#         print(f"AI: {response}")


# while True:
#     query = st.chat_input("You: ", key=str(uuid4()))  # Add unique key
#     if query:
#         st.chat_message("User").write(f"You: {query}")
#         if query.lower() == "exit":
#             break
        
#         state = graph.invoke({"messages": [query]})
#         response = state["messages"][-1].content
#         st.chat_message("AI").write(f"AI: {response}")