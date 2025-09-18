
"""
MCP (Model Context Protocol): Defines message structure, broker, and constants for agent communication.

Production Notes:
- Provides in-memory message broker (MCPBroker) for agent-to-agent communication.
- Defines message types, agent IDs, and message creation helpers.
- Used by all agents for decoupled, event-driven workflows.

Future Scope:
- Add persistent message queue (e.g., Redis, RabbitMQ) for distributed agents.
- Support message replay, auditing, and delivery guarantees.
- Add message schemas for validation and versioning.
"""
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MCPMessage(BaseModel):
    """
    Base MCP message structure for agent communication.
    Used for all agent-to-agent messages in the system.
    """
    sender: str
    receiver: str
    type: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any]

    class Config:
        frozen = True


from utils.logger import Logger

class MCPBroker:
    """
    In-memory message broker for agent communication.
    Handles agent subscription, message publishing, and message history.
    """
    def __init__(self):
        self._subscribers = {}
        self._message_history = []
        self._logger = Logger("MCPBroker")

    def subscribe(self, agent_id: str, callback):
        self._logger.info(f"Subscribing agent {agent_id} to MCP broker")
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)
        self._logger.info(f"Agent {agent_id} subscribed. Total subscribers: {len(self._subscribers[agent_id])}")

    def publish(self, message: MCPMessage):
        self._logger.info(f"Publishing message to {message.receiver}: type={message.type}, trace_id={message.trace_id}")
        self._message_history.append(message)
        if message.receiver in self._subscribers:
            for i, callback in enumerate(self._subscribers[message.receiver]):
                self._logger.info(f"Invoking callback {i+1} for agent {message.receiver}")
                callback(message)
            self._logger.info(f"All callbacks for {message.receiver} invoked.")
        else:
            self._logger.warning(f"No subscribers found for receiver {message.receiver}")
        return message.trace_id
    
    def get_message_history(self, trace_id: Optional[str] = None) -> List[MCPMessage]:
        """
        Get message history, optionally filtered by trace_id.
        Useful for debugging and tracing workflows.
        """
        if trace_id:
            return [msg for msg in self._message_history if msg.trace_id == trace_id]
        return self._message_history


# Singleton instance of the MCP broker
mcp_broker = MCPBroker()


# Message type constants
class MessageType:
    """
    Message type constants for agent communication events.
    """
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    DOCUMENT_PROCESSED = "DOCUMENT_PROCESSED"
    QUERY = "QUERY"
    CONTEXT_REQUEST = "CONTEXT_REQUEST"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    CHAT_HISTORY_REQUEST = "CHAT_HISTORY_REQUEST"
    CHAT_HISTORY_RESPONSE = "CHAT_HISTORY_RESPONSE"
    ERROR = "ERROR"


# Agent ID constants
class AgentID:
    """
    Agent ID constants for identifying agent roles in the system.
    """
    INGESTION = "IngestionAgent"
    RETRIEVAL = "RetrievalAgent"
    CHAT = "ChatAgent"
    LLM = "LLMResponseAgent"
    UI = "UIAgent"


def create_mcp_message(
    sender: str,
    receiver: str,
    msg_type: str,
    payload: Dict[str, Any],
    trace_id: Optional[str] = None
) -> MCPMessage:
    """
    Helper function to create MCP messages with or without a trace_id.
    """
    if trace_id:
        return MCPMessage(
            sender=sender,
            receiver=receiver,
            type=msg_type,
            trace_id=trace_id,
            payload=payload
        )
    return MCPMessage(
        sender=sender,
        receiver=receiver,
        type=msg_type,
        payload=payload
    )