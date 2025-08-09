# Configuration settings for the project
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_VERSION = "v1"
TIMEOUT = 180  # seconds

# LangChain Configuration
# Can use open API directly as well if needed.
OPENAI_API_KEY = os.getenv("AI_PIPE_TOKEN")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")  # Added for Gemini
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Model Configuration
# DEFAULT_OPENAI_MODEL = "gpt-4"
# Upgraded to GPT-4 for better accuracy can be done
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Updated to Gemini 1.5 Pro for enhanced capabilities
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"
TEMPERATURE = 0.3  # Lower temperature for more accurate analysis
MAX_TOKENS = 4000

# Vector Store Configuration
VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
EMBEDDING_MODEL = "text-embedding-ada-002"  # Reverted for stability

# Chain Configuration
CHAIN_TIMEOUT = 120  # seconds
MAX_RETRIES = 3


def get_chat_model(provider="openai"):
    """
    Get a chat model instance with configuration based on the provider.
    :param provider: 'openai' or 'gemini'
    """
    if provider == "openai":
        try:
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                base_url="https://aipipe.org/openrouter/v1",
                model_name=DEFAULT_OPENAI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
        except ImportError:
            # Fallback for newer LangChain versions
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=OPENAI_API_KEY,
                    model=DEFAULT_OPENAI_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
            except ImportError:
                raise ImportError(
                    "Could not import ChatOpenAI from langchain or langchain_openai")
    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=DEFAULT_GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
                convert_system_message_to_human=True  # Important for some chains
            )
        except ImportError:
            raise ImportError(
                "Could not import ChatGoogleGenerativeAI. Please install langchain-google-genai.")
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Choose 'openai' or 'gemini'.")
