
"""
BaseAgent: Abstract base class for all agents in the agentic RAG chatbot system.

Production Notes:
- Provides message handler registration, message routing, and logging.
- All agents inherit from this class and implement the start() method.
- Integrates with MCPBroker for in-memory message passing.

Future Scope:
- Add agent lifecycle hooks (shutdown, restart, health checks).
- Support for distributed or remote agent communication.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from models.mcp import MCPMessage, mcp_broker
from utils.logger import Logger



class BaseAgent(ABC):
    _logger = Logger("BaseAgent")
    """
    Abstract base class for all agents.
    Handles:
      - Message handler registration and routing
      - Logging and MCPBroker integration
    """

    def __init__(self, agent_id: str):
        self._logger.info(f"BaseAgent.__init__ called with agent_id={agent_id}")
        self.agent_id = agent_id
        self.logger = Logger(agent_id)
        self._callbacks = {}
        mcp_broker.subscribe(agent_id, self._handle_message)

    def _handle_message(self, message: MCPMessage):
        self._logger.info(f"_handle_message called with message: {message}")
        """
        Route incoming MCP messages to registered handlers.
        """
        self.logger.log_mcp_message(message.dict(), direction="received")
        if message.type in self._callbacks:
            for callback in self._callbacks[message.type]:
                callback(message)
        else:
            self.logger.warning(f"No handler for message type: {message.type}")

    def register_handler(self, message_type: str, callback: Callable[[MCPMessage], None]):
        self._logger.info(f"register_handler called for message_type: {message_type}")
        """
        Register a callback for a specific message type.
        """
        if message_type not in self._callbacks:
            self._callbacks[message_type] = []
        self._callbacks[message_type].append(callback)

    def send_message(self, receiver: str, msg_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None) -> str:
        self._logger.info(f"send_message called: receiver={receiver}, msg_type={msg_type}, trace_id={trace_id}")
        """
        Send an MCP message to another agent via the broker.
        """
        from models.mcp import create_mcp_message
        message = create_mcp_message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            trace_id=trace_id
        )
        self.logger.log_mcp_message(message.dict(), direction="sent")
        return mcp_broker.publish(message)

    @abstractmethod
    def start(self):
        self._logger.info("start called on BaseAgent")
        """
        Start the agent (must be implemented by subclasses).
        """
        pass