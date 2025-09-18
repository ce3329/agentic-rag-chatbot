
"""
RetrievalAgent: Handles vector store integration and semantic search for the agentic RAG chatbot.

Production Notes:
- Integrates with Pinecone for vector storage and retrieval.
- Encodes document chunks and queries using SentenceTransformers.
- Handles document upsert and semantic search for context retrieval.

Future Scope:
- Add support for hybrid search (keyword + vector).
- Implement advanced filtering, ranking, and metadata search.
- Add support for multiple vector DBs or local fallback.
"""
import os
from typing import Any, Dict, List

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from agents.base_agent import BaseAgent
from config.settings import (
    EMBEDDING_MODEL, PINECONE_API_KEY, PINECONE_ENVIRONMENT, 
    PINECONE_INDEX_NAME, SIMILARITY_THRESHOLD, TOP_K_RESULTS
)
from models.mcp import AgentID, MessageType
from utils.logger import Logger



class RetrievalAgent(BaseAgent):
    _logger = Logger("RetrievalAgent")
    """
    Agent responsible for vector store operations and semantic search.
    Handles:
      - Storing document embeddings in Pinecone
      - Semantic search for context retrieval
      - Routing context to LLM agent
    """

    def __init__(self):
        self._logger.info("RetrievalAgent.__init__ called")
        super().__init__(AgentID.RETRIEVAL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        try:
            self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
            indexes = self.pc.list_indexes()
            index_exists = any(index.name == PINECONE_INDEX_NAME for index in indexes)
            if not index_exists:
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=self.embedding_model.get_sentence_embedding_dimension(),
                    metric="cosine"
                )
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            self.logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone: {str(e)}")
            self.index = None
        # Register handlers for document processed and context request
        self.register_handler(MessageType.DOCUMENT_PROCESSED, self._handle_document_processed)
        self.register_handler(MessageType.CONTEXT_REQUEST, self._handle_context_request)

    def start(self):
        self._logger.info("RetrievalAgent.start called")
        """Start the retrieval agent."""
        self.logger.info("RetrievalAgent started")

    def _handle_document_processed(self, message):
        self._logger.info(f"_handle_document_processed called with message: {message}")
        """
        Receives processed document chunks and embeddings, upserts to Pinecone.
        """
        payload = message.payload
        chunks = payload.get("chunks", [])
        embeddings = payload.get("embeddings", [])
        trace_id = message.trace_id
        if not chunks or not embeddings:
            self.logger.error("Missing chunks or embeddings in document processed message")
            return
        if self.index is None:
            self.logger.warning("Pinecone index not available, skipping document processing")
            return
        self.logger.info(f"Storing {len(chunks)} chunks in vector store", trace_id=trace_id)
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{trace_id}-{i}"
            vector = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "trace_id": trace_id
                }
            }
            vectors.append(vector)
        print(f"Embedding {len(vectors)} vectors to Pinecone...")
        try:
            self.index.upsert(vectors=vectors)
            self.logger.info(f"Successfully stored {len(vectors)} vectors in Pinecone")
        except Exception as e:
            self.logger.error(f"Failed to upsert vectors to Pinecone: {str(e)}")

    def _handle_context_request(self, message):
        self._logger.info(f"_handle_context_request called with message: {message}")
        """
        Receives a user query, performs semantic search, and sends context to LLM agent.
        """
        payload = message.payload
        query = payload.get("query", "")
        trace_id = message.trace_id
        self.logger.info(f"Received context request for query: {query}", trace_id=trace_id)
        if not query:
            self.logger.error("Missing query in context request message")
            return
        if self.index is None:
            self.logger.warning("Pinecone index not available, returning empty results")
            self.logger.info("Sending empty context response due to missing index", trace_id=trace_id)
            self.send_message(
                receiver=AgentID.LLM,
                msg_type=MessageType.CONTEXT_RESPONSE,
                payload={"context": [], "query": query},
                trace_id=trace_id
            )
            return
        self.logger.info(f"Processing context request for query: {query}", trace_id=trace_id)
        try:
            self.logger.info("Generating embedding for query", trace_id=trace_id)
            query_embedding = self.embedding_model.encode(query).tolist()
            self.logger.info("Querying Pinecone vector store", trace_id=trace_id)
            results = self.index.query(
                vector=query_embedding,
                top_k=TOP_K_RESULTS,
                include_metadata=True
            )
            self.logger.info(f"Retrieved {len(results.matches) if hasattr(results, 'matches') else 0} results from vector store", trace_id=trace_id)
        except Exception as e:
            self.logger.error(f"Failed to query Pinecone: {str(e)}")
            self.logger.info("Sending empty context response due to query error", trace_id=trace_id)
            self.send_message(
                receiver=AgentID.LLM,
                msg_type=MessageType.CONTEXT_RESPONSE,
                payload={"context": [], "query": query},
                trace_id=trace_id
            )
            return
        filtered_results = [
            match for match in results.matches 
            if match.score >= SIMILARITY_THRESHOLD
        ]
        # Future: Remove print, add structured logging for match metadata if needed
        chunks = [match.metadata.get("text", "") for match in filtered_results]
        sources = [match.metadata.get("source", "") for match in filtered_results]
        self.logger.info(
            f"Retrieved {len(chunks)} relevant chunks for query", 
            trace_id=trace_id,
            threshold=SIMILARITY_THRESHOLD
        )
        self.logger.info(f"Sending context response with {len(chunks)} chunks to LLM agent", trace_id=trace_id)
        self.send_message(
            receiver=AgentID.LLM,
            msg_type=MessageType.CONTEXT_RESPONSE,
            payload={
                "query": query,
                "chunks": chunks,
                "sources": sources
            },
            trace_id=trace_id
        )