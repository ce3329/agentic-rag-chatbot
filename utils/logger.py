
"""
Logger: Structured logging system for the agentic RAG chatbot.

Production Notes:
- Uses structlog for structured, JSON-formatted logs.
- Centralizes logging configuration for all agents and app modules.
- Supports info, error, warning, debug, and critical levels.

Future Scope:
- Add support for remote log aggregation (e.g., ELK, Datadog).
- Integrate trace IDs and session IDs for distributed tracing.
- Add log rotation and retention policies.
"""
import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from config.settings import LOG_LEVEL


def configure_logging():
    """
    Configure structured logging for the application using structlog and stdlib.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, LOG_LEVEL)
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **initial_values: Any):
    """
    Get a structured logger with the given name and initial values.
    """
    return structlog.get_logger(name, **initial_values)


# Configure logging on module import
configure_logging()



class Logger:
    """
    Logger class for structured logging.
    Wraps structlog for consistent log formatting and usage across agents.
    """

    def __init__(self, name: str, **initial_values: Any):
        self.logger = get_logger(name, **initial_values)

    def info(self, message: str, **kwargs: Any):
        """Log an info message."""
        self.logger.info(message, **kwargs)

    def error(self, message: str, **kwargs: Any):
        """Log an error message."""
        self.logger.error(message, **kwargs)

    def warning(self, message: str, **kwargs: Any):
        """Log a warning message."""
        self.logger.warning(message, **kwargs)

    def debug(self, message: str, **kwargs: Any):
        """Log a debug message."""
        self.logger.debug(message, **kwargs)

    def critical(self, message: str, **kwargs: Any):
        """Log a critical message."""
        self.logger.critical(message, **kwargs)

    def log_mcp_message(self, message_dict: Dict[str, Any], direction: str = "sent"):
        """
        Log an MCP message (for agent communication tracing).
        """
        self.logger.info(
            f"MCP message {direction}",
            sender=message_dict.get("sender"),
            receiver=message_dict.get("receiver"),
            type=message_dict.get("type"),
            trace_id=message_dict.get("trace_id"),
        )