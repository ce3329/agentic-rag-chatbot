
"""
LLMResponseAgent: Handles LLM-based response generation for the agentic RAG chatbot.

Production Notes:
- Integrates with Google Gemini and Groq LLMs for response generation.
- Receives context from RetrievalAgent, prepares prompts, and sends answers to ChatAgent and UI.
- Designed for extensibility: add more LLM providers or prompt strategies as needed.

Future Scope:
- Add support for multi-provider fallback and dynamic provider selection.
- Implement streaming/token-by-token LLM responses for real-time UI.
- Enhance prompt engineering and context window management.
"""


import google.generativeai as genai
import groq

from agents.base_agent import BaseAgent
from config.settings import DEFAULT_LLM_PROVIDER, FALLBACK_LLM_PROVIDER, GEMINI_API_KEY, GROQ_API_KEY
from models.mcp import AgentID, MessageType
from utils.logger import Logger



class LLMResponseAgent(BaseAgent):
    _logger = Logger("LLMResponseAgent")
    """
    Agent responsible for generating LLM responses.
    Handles:
      - Receiving context from RetrievalAgent
      - Prompt construction and LLM invocation
      - Sending answers to ChatAgent and UI
    """

    def __init__(self):
        self._logger.info("LLMResponseAgent.__init__ called")
        super().__init__(AgentID.LLM)
        self._init_gemini()
        self._init_groq()
        # Register handler for context responses from RetrievalAgent
        self.register_handler(MessageType.CONTEXT_RESPONSE, self._handle_context_response)

    def start(self):
        self._logger.info("LLMResponseAgent.start called")
        """Start the LLM response agent."""
        self.logger.info("LLMResponseAgent started")

    def _init_gemini(self):
        self._logger.info("_init_gemini called")
        """Configure Gemini API client (Google)."""
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')

    def _init_groq(self):
        self._logger.info("_init_groq called")
        """Configure Groq API client (future: fallback or multi-provider)."""
        try:
            self.groq_client = groq.Client(api_key=GROQ_API_KEY)
            self.logger.info("Successfully initialized Groq client")
        except TypeError:
            self.groq_client = None
            self.logger.warning("Failed to initialize Groq client, will use fallback provider")
            self.logger.error("Failed to initialize Groq client due to API incompatibility")

    def _handle_context_response(self, message):
        self._logger.info(f"_handle_context_response called with message: {message}")
        """
        Main entrypoint: receives context from RetrievalAgent, prepares prompt, calls LLM, and routes answer.
        Workflow:
          1. Receives context (chunks, sources) for a user query.
          2. Optionally requests chat history (future: use for conversational memory).
          3. Prepares prompt and calls Gemini LLM.
          4. Sends answer to ChatAgent (for storage) and UI (for display).
        """
        payload = message.payload
        query = payload.get("query", "")
        chunks = payload.get("chunks", [])
        sources = payload.get("sources", [])
        trace_id = message.trace_id

        self.logger.info(f"Received context response for query: {query}", trace_id=trace_id)
        self.logger.info(f"Context contains {len(chunks)} chunks and {len(sources)} sources", trace_id=trace_id)
        self.logger.info(f"Generating response for query: {query}", trace_id=trace_id)

        # Future: Use chat history for conversational context
        self.logger.info("Requesting chat history from ChatAgent", trace_id=trace_id)
        self.send_message(
            receiver=AgentID.CHAT,
            msg_type=MessageType.CHAT_HISTORY_REQUEST,
            payload={"session_id": trace_id},
            trace_id=trace_id
        )

        prompt = self._prepare_prompt(query, chunks)

        # Only Gemini is used for now; future: add fallback/multi-provider
        import traceback
        try:
            self.logger.info("Calling Gemini LLM API...")
            response = self._generate_response(prompt, "gemini")
            self.logger.info("Gemini LLM API call successful.")
        except Exception as e:
            self.logger.error(f"Error with Gemini LLM provider: {str(e)}")
            print("[LLMResponseAgent] Gemini Exception:", e)
            traceback.print_exc()
            response = "[LLM Error: Unable to generate response]"

        # Route answer to ChatAgent (for storage) and UI (for display)
        self.send_message(
            receiver=AgentID.CHAT,
            msg_type=MessageType.LLM_RESPONSE,
            payload={
                "response": response,
                "session_id": trace_id,
                "sources": sources
            },
            trace_id=trace_id
        )
        self.send_message(
            receiver=AgentID.UI,
            msg_type=MessageType.LLM_RESPONSE,
            payload={
                "response": response,
                "sources": sources
            },
            trace_id=trace_id
        )
    

    def _prepare_prompt(self, query, chunks):
        self._logger.info(f"_prepare_prompt called: query={query}, chunks_count={len(chunks) if chunks else 0}")
        """
        Prepare a context-grounded prompt for the LLM.
        - If no context, instruct LLM to say it cannot answer.
        - If context, instruct LLM to answer strictly from context, referencing chunks.
        """
        if not chunks or all(not c.strip() for c in chunks):
            prompt = f"""
            You are a helpful assistant that answers questions based ONLY on the provided context.\n
            There is no relevant information in the context for this question.\n
            CONTEXT:\n
            USER QUESTION:\n{query}\n
            ANSWER:\nI don't have enough information to answer this question.\n
            """
        else:
            context = "\n\n".join(chunks)
            prompt = f"""
            You are a precise and reliable assistant. Answer the user's question using ONLY the information from the provided context.

            Instructions:
            - If multiple context chunks are relevant, synthesize them into a clear, well-structured answer.
            - Ignore irrelevant or repetitive context.
            - Do not invent or assume facts beyond the context.
            - If the context does not provide enough information, reply exactly: "I don’t have enough information to answer this question."
            - Keep the answer concise, coherent, and factually grounded.
            - When using context, explicitly reference the chunk(s) you relied on (e.g., [Chunk 2], [Chunk 5]).

            CONTEXT (each chunk is labeled for reference):
            {context}

            USER QUESTION:
            {query}

            ANSWER:
            (Provide a clear, concise, context-grounded answer. Reference the chunks inline where relevant, e.g., "The report states [Chunk 3].")

            CHUNKS USED:
            (List the IDs of all chunks used in the answer, separated by commas. If none are used, write "None".)
            """
        return prompt

    
    def _generate_response(self, prompt, provider):
        self._logger.info(f"_generate_response called: provider={provider}")
        """
        Call the LLM provider to generate a response.
        Currently only Gemini is supported. Future: add provider switch/fallback.
        """
        response = self.gemini_model.generate_content(prompt)
        return response.text