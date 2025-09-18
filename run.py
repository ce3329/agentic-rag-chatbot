
"""
Run script to start the agentic RAG chatbot (CLI entrypoint).

Production Notes:
- Initializes all agents and launches Streamlit UI in a subprocess/thread.
- Used for local development and quickstart.

Future Scope:
- Add CLI options for headless/server mode, agent selection, and config overrides.
- Support for process monitoring and graceful shutdown.
"""
import os
import sys
import subprocess
import threading

# Add detailed logging for tracing

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import initialize_agents
from utils.logger import Logger


def run_streamlit():
    logger.info("Preparing to launch Streamlit UI...")
    streamlit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py")
    logger.info(f"Streamlit path resolved: {streamlit_path}")
    try:
        logger.info("Running Streamlit subprocess...")
        subprocess.run(["streamlit", "run", streamlit_path])
        logger.info("Streamlit subprocess finished.")
    except Exception as e:
        logger.error(f"Exception occurred while running Streamlit: {e}")


if __name__ == "__main__":
    logger = Logger("RunScript")
    logger.info("Run script started.")
    try:
        # Initialize agents
        logger.info("Initializing agents...")
        agents = initialize_agents()
        logger.info("Agents initialized successfully")
    except Exception as e:
        logger.error(f"Exception during agent initialization: {e}")
        sys.exit(1)

    try:
        # Start Streamlit in a separate thread
        logger.info("Starting Streamlit UI thread...")
        streamlit_thread = threading.Thread(target=run_streamlit)
        streamlit_thread.daemon = True
        streamlit_thread.start()
        logger.info("Streamlit UI thread started.")
    except Exception as e:
        logger.error(f"Exception while starting Streamlit thread: {e}")
        sys.exit(1)

    logger.info("Agentic RAG Chatbot is running. Press Ctrl+C to exit.")

    try:
        # Keep the main thread alive
        logger.info("Main thread waiting for Streamlit thread to finish...")
        streamlit_thread.join()
        logger.info("Streamlit thread finished.")
    except KeyboardInterrupt:
        logger.info("Shutting down Agentic RAG Chatbot")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exception in main thread: {e}")
        sys.exit(1)