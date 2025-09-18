
"""
ChatAgent: Manages conversation history for the agentic RAG chatbot.

Production Notes:
- Stores and retrieves user/assistant messages in MongoDB for each session.
- Handles user queries, LLM responses, and chat history requests.
- Forwards queries to RetrievalAgent and routes LLM responses to storage.

Future Scope:
- Add support for multi-session, multi-user chat.
- Integrate advanced conversation memory (summarization, search).
- Add message deletion, editing, and export features.
"""

from typing import Dict, List, Optional
from utils.logger import Logger
from datetime import datetime

import pymongo
from pymongo import MongoClient

from agents.base_agent import BaseAgent
from config.settings import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION
from models.mcp import AgentID, MessageType



class ChatAgent(BaseAgent):
    _logger = Logger("ChatAgent")
    """
    Agent responsible for managing conversation history.
    Handles:
      - Storing user and assistant messages in MongoDB
      - Forwarding user queries to RetrievalAgent
      - Routing LLM responses to storage
      - Serving chat history requests
    """

    def __init__(self):
        self._logger.info("ChatAgent.__init__ called")
        super().__init__(AgentID.CHAT)
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[MONGODB_DB_NAME]
        self.collection = self.db[MONGODB_COLLECTION]
        # Register handlers for query, LLM response, and chat history
        self.register_handler(MessageType.QUERY, self._handle_query)
        self.register_handler(MessageType.LLM_RESPONSE, self._handle_llm_response)
        self.register_handler(MessageType.CHAT_HISTORY_REQUEST, self._handle_chat_history_request)

    def start(self):
        self._logger.info("ChatAgent.start called")
        """Start the chat agent."""
        self.logger.info("ChatAgent started")

    def _handle_query(self, message):
        self._logger.info(f"_handle_query called with message: {message}")
        """
        Receives user query, stores it, and forwards to RetrievalAgent for context.
        """
        payload = message.payload
        query = payload.get("query", "")
        session_id = payload.get("session_id", "default")
        trace_id = message.trace_id
        self.logger.info(f"Received user query: {query}", trace_id=trace_id)
        self._store_message(
            session_id=session_id,
            trace_id=trace_id,
            role="user",
            content=query
        )
        self.logger.info(f"Sending context request to retrieval agent for query: {query}", trace_id=trace_id)
        self.send_message(
            receiver=AgentID.RETRIEVAL,
            msg_type=MessageType.CONTEXT_REQUEST,
            payload={
                "query": query,
                "session_id": session_id
            },
            trace_id=trace_id
        )
        self.logger.info(f"Context request sent to retrieval agent", trace_id=trace_id)

    def _handle_llm_response(self, message):
        self._logger.info(f"_handle_llm_response called with message: {message}")
        """
        Receives LLM response, stores assistant message in conversation history.
        """
        payload = message.payload
        response = payload.get("response", "")
        session_id = payload.get("session_id", "default")
        sources = payload.get("sources", [])
        trace_id = message.trace_id
        self._store_message(
            session_id=session_id,
            trace_id=trace_id,
            role="assistant",
            content=response,
            sources=sources
        )

    def _handle_chat_history_request(self, message):
        self._logger.info(f"_handle_chat_history_request called with message: {message}")
        """
        Handles requests for chat history (for UI or LLM context).
        """
        payload = message.payload
        session_id = payload.get("session_id", "default")
        limit = payload.get("limit", 10)
        trace_id = message.trace_id
        history = self.get_conversation_history(session_id, limit)
        self.send_message(
            receiver=message.sender,
            msg_type=MessageType.CHAT_HISTORY_RESPONSE,
            payload={
                "session_id": session_id,
                "history": history
            },
            trace_id=trace_id
        )

    def _store_message(self, session_id: str, trace_id: str, role: str, content: str, sources: Optional[List[str]] = None):
        self._logger.info(f"_store_message called: session_id={session_id}, trace_id={trace_id}, role={role}")
        """
        Store a message (user or assistant) in MongoDB for the session.
        """
        message_doc = {
            "session_id": session_id,
            "trace_id": trace_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "sources": sources or []
        }
        self.collection.insert_one(message_doc)
        self.logger.info(f"Stored {role} message in conversation history", session_id=session_id)

    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        self._logger.info(f"get_conversation_history called: session_id={session_id}, limit={limit}")
        """
        Retrieve the most recent messages for a session (for UI or LLM context).
        """
        cursor = self.collection.find(
            {"session_id": session_id}
        ).sort("timestamp", pymongo.DESCENDING).limit(limit)
        history = list(cursor)
        history.reverse()
        for item in history:
            if "_id" in item:
                del item["_id"]
        return history