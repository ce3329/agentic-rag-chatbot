
"""
Main entrypoint for agent initialization and orchestration in the agentic RAG chatbot.

Production Notes:
- Initializes all agents and starts them in-process.
- Used for CLI or script-based orchestration (not Streamlit UI).

Future Scope:
- Add graceful shutdown and restart logic for agents.
- Support for running agents in separate processes or containers.
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.chat_agent import ChatAgent
from agents.llm_response_agent import LLMResponseAgent
from utils.logger import Logger


def initialize_agents():
    """Initialize all agents in the system."""
    logger = Logger("Main")
    logger.info("Initializing agents")
    
    # Initialize agents
    ingestion_agent = IngestionAgent()
    retrieval_agent = RetrievalAgent()
    chat_agent = ChatAgent()
    llm_agent = LLMResponseAgent()
    
    # Start agents
    ingestion_agent.start()
    retrieval_agent.start()
    chat_agent.start()
    llm_agent.start()
    
    logger.info("All agents initialized and started")
    
    return {
        "ingestion_agent": ingestion_agent,
        "retrieval_agent": retrieval_agent,
        "chat_agent": chat_agent,
        "llm_agent": llm_agent
    }


if __name__ == "__main__":
    # Initialize agents
    agents = initialize_agents()
    
    # Keep the application running
    try:
        logger = Logger("Main")
        logger.info("Agentic RAG Chatbot is running. Press Ctrl+C to exit.")
        
        # This will keep the script running until interrupted
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Agentic RAG Chatbot")