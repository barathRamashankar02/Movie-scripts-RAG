# Movie-scripts-RAG
Conversational Chat Application with Movie Script Retrieval
Overview
This project is a conversational chat application that leverages vector databases to store and retrieve movie scripts, enabling it to answer questions contextually based on the stored content. Additionally, it features general conversational capabilities, allowing users to interact beyond script-based queries.

Features
Movie Script-Based Question Answering: Stores and retrieves movie scripts from a vector database to provide relevant responses.
Conversational Chat System: Implements a message state graph using Langraph, making interactions more dynamic and allowing responses beyond database-dependent queries.
Two Chat Modes:
Simple Chat System: Answers one question at a time without remembering chat history.
Interactive Chat System: Maintains chat history and handles general conversations using MemorySaver Checkpointer in Langraph.
Efficient PDF Processing: Uses PyPDF2 to extract text from movie script PDFs.
Optimized Text Chunking: Leverages LangChain’s Recursive Text Splitter to manage and structure script content.
Web Interface: Built with Gradio for a user-friendly experience.
Installation
Clone the repository:
bash
Copy
Edit
git clone <repository-url>
cd <repository-folder>
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Set up API keys:
Create a .env file in the root directory.
Add necessary API keys (default: OpenAI API Key). Adjust the code if using a different LLM provider.
ini
Copy
Edit
OPENAI_API_KEY=your-api-key-here
Usage
Run the application:
bash
Copy
Edit
python gradio_app.py
The chat application will be accessible via localhost in your web browser.
Technologies Used
Python
LangChain (for text processing & vector database management)
Langraph (for conversational message state graph)
PyPDF2 (for extracting text from PDFs)
Gradio (for web-based UI)
Vector Database (for storing & retrieving movie scripts)
Contributing
Feel free to submit issues, pull requests, or feature suggestions to improve the chatbot!
