
"""
Streamlit UI for the agentic RAG chatbot.

Production Notes:
- Provides chat interface, document upload, and real-time agent interaction.
- Uses session state to manage chat, responses, and agent initialization.
- UIAgent subscribes to MCPBroker for LLM responses.

Future Scope:
- Add authentication, user management, and multi-session support.
- Improve UI/UX with advanced chat features and document previews.
- Support for streaming LLM responses and real-time updates.
"""
import os
import sys
import uuid
import tempfile
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import streamlit as st
from app.main import initialize_agents
from utils.logger import Logger

from models.mcp import AgentID, MessageType, create_mcp_message, mcp_broker
from config.settings import SUPPORTED_DOCUMENT_TYPES


class UIAgent:
    """UI Agent for handling user interactions."""
    def __init__(self):
        self.logger = Logger("UIAgent")
        self.agent_id = AgentID.UI
        self.logger.info("UIAgent initialized and subscribing to MCP broker")
        # Register with MCP broker
        mcp_broker.subscribe(self.agent_id, self._handle_message)

    def _handle_message(self, message):
        self.logger.info(f"UIAgent._handle_message called with message type: {message.type}")
        if message.type == MessageType.LLM_RESPONSE:
            self.logger.info("Received LLM_RESPONSE, updating session state")
            # Store the response in session state
            st.session_state.responses.append({
                "text": message.payload.get("response", ""),
                "sources": message.payload.get("sources", [])
            })
            st.session_state.waiting_for_response = False
            self.logger.info("Session state updated, waiting_for_response set to False")


# Initialize session state
def init_session_state():
    logger = Logger("StreamlitApp")
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"Initialized session_id: {st.session_state.session_id}")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        logger.info("Initialized messages list in session state")
    if "responses" not in st.session_state:
        st.session_state.responses = []
        logger.info("Initialized responses list in session state")
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
        logger.info("Initialized waiting_for_response in session state")


def main():
    # Initialize all agents in the same process as Streamlit UI
    if 'agents_initialized' not in st.session_state:
        initialize_agents()
        st.session_state['agents_initialized'] = True
    logger = Logger("StreamlitApp")
    st.set_page_config(
        page_title="Agentic RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    logger.info("Session state initialized")
    # Initialize UI Agent
    ui_agent = UIAgent()
    logger.info("UIAgent initialized")
    
    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=SUPPORTED_DOCUMENT_TYPES
        )
        if uploaded_files:
            logger.info(f"{len(uploaded_files)} files uploaded in sidebar")
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                        logger.info(f"Saved uploaded file: {file_path}")
                    message = create_mcp_message(
                        sender=AgentID.UI,
                        receiver=AgentID.INGESTION,
                        msg_type=MessageType.DOCUMENT_UPLOAD,
                        payload={"file_paths": file_paths}
                    )
                    logger.info(f"Publishing DOCUMENT_UPLOAD message to MCP broker: {message}")
                    mcp_broker.publish(message)
                    st.success(f"Successfully processed {len(file_paths)} documents!")
    
    # Main chat interface
    st.title("Agentic RAG Chatbot")
    st.caption("Ask questions about your uploaded documents")
    
    # Display chat messages
    for i, (msg, resp) in enumerate(zip(st.session_state.messages, st.session_state.responses)):
        with st.chat_message("user"):
            st.write(msg)
        
        with st.chat_message("assistant"):
            st.write(resp["text"])
            
            if resp["sources"]:
                with st.expander("Sources"):
                    for source in resp["sources"]:
                        st.write(f"- {os.path.basename(source)}")
    
    # Handle remaining messages without responses
    for i in range(len(st.session_state.responses), len(st.session_state.messages)):
        with st.chat_message("user"):
            st.write(st.session_state.messages[i])
    
    # Chat input
    if st.session_state.waiting_for_response:
        logger.info("UI waiting for response from backend agent...")
        st.info("Waiting for response...")
    else:
        prompt = st.chat_input("Ask a question about your documents")
        if prompt:
            logger.info(f"User submitted prompt: {prompt}")
            st.session_state.messages.append(prompt)
            st.session_state.waiting_for_response = True
            logger.info("Set waiting_for_response to True after user prompt")
            message = create_mcp_message(
                sender=AgentID.UI,
                receiver=AgentID.CHAT,
                msg_type=MessageType.QUERY,
                payload={
                    "query": prompt,
                    "session_id": st.session_state.session_id
                }
            )
            logger.info(f"Publishing QUERY message to MCP broker: {message}")
            mcp_broker.publish(message)
            st.rerun()


if __name__ == "__main__":
    main()