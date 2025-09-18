
"""
Centralized configuration for the agentic RAG chatbot.

Production Notes:
- Stores all environment, model, vector DB, and LLM config in one place.
- Used by all agents and app modules for consistent settings.

Future Scope:
- Add config validation and schema enforcement.
- Support for environment-specific overrides (dev, prod, test).
- Integrate with secret managers for sensitive keys.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Vector Store & Embeddings Configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.35
BATCH_SIZE = 100

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "rag_chatbot")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "conversations")

# LLM Configuration
DEFAULT_LLM_PROVIDER = "gemini"
FALLBACK_LLM_PROVIDER = "groq"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Session Security
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "default_insecure_key")

# Supported Document Types
SUPPORTED_DOCUMENT_TYPES = [
    "pdf", "docx", "pptx", "csv", "txt", "md"
]

# Chunking Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200