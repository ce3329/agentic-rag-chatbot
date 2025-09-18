
# Agentic RAG Chatbot

An agent-based Retrieval-Augmented Generation (RAG) chatbot that answers user questions strictly based on user-uploaded documents. Modular, production-ready, and designed for extensibility.


## Features

- **Document Ingestion & Parsing**: PDF, PPTX, CSV, DOCX, TXT, Markdown
- **Agentic Architecture**: Specialized agents for ingestion, retrieval, chat, and LLM response
- **Model Context Protocol (MCP)**: Structured, decoupled agent communication
- **Vector Store & Embeddings**: Pinecone + SentenceTransformers for semantic search
- **Multi-turn Conversations**: MongoDB-backed chat history
- **Source References**: Every answer cites its source


## Architecture & Workflow

**Agents:**
- `IngestionAgent`: Parses and chunks uploaded documents, generates embeddings
- `RetrievalAgent`: Stores embeddings in Pinecone, performs semantic search
- `ChatAgent`: Manages conversation history in MongoDB, routes queries and responses
- `LLMResponseAgent`: Prepares prompts and generates answers using Gemini (or Groq)

**Workflow:**
1. User uploads documents via Streamlit UI
2. IngestionAgent processes and chunks documents, sends to RetrievalAgent
3. RetrievalAgent stores vectors, handles semantic search for queries
4. ChatAgent records all messages, manages session history
5. LLMResponseAgent generates answers using retrieved context and LLM


## Setup

### Prerequisites
- Python 3.11+
- MongoDB (local or cloud)
- Pinecone account
- Gemini and Groq API keys

### Installation
1. Clone this repository:
   ```
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=rag-chatbot
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB_NAME=rag_chatbot
   MONGODB_COLLECTION=conversations
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   LOG_LEVEL=INFO
   ```


## Usage

1. Start the chatbot:
   ```
   streamlit run app/streamlit_app.py 
   ```
2. Open the Streamlit UI (URL will be shown in the terminal)
3. Upload documents and chat!


## Configuration

All settings are in `config/settings.py` and `.env`.
- Vector DB, embedding model, chunking, and LLM provider can be changed here.

## GitHub Repository Usage

1. [Create a new GitHub repository](https://github.com/new) (no README or .gitignore)
2. Add your remote and push:
   ```
   git remote add origin <your-repo-url>
   git push -u origin master
   ```

## Contributing

Pull requests and issues are welcome! Please open an issue for bugs, feature requests, or questions.
\

