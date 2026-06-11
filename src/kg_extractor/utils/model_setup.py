"""Shared API key management utilities."""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

PARSING_PROVIDER = os.getenv("PARSING_PROVIDER", "nvidia")
CHUNKING_PROVIDER = os.getenv("CHUNKING_PROVIDER", "openai")
TRIPLET_PROVIDER = os.getenv("TRIPLET_PROVIDER", "openai")
REFINEMENT_PROVIDER = os.getenv("REFINEMENT_PROVIDER", "openai")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

PARSING_MODEL = os.getenv("PARSING_MODEL", "microsoft/phi-4-multimodal-instruct")
CHUNKING_MODEL = os.getenv("CHUNKING_MODEL", "gpt-4o-mini")
TRIPLET_MODEL = os.getenv("TRIPLET_MODEL", "gpt-4o-mini")
REFINEMENT_MODEL = os.getenv("REFINEMENT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EVIDENCE_EMBEDDING_MODEL = os.getenv("EVIDENCE_EMBEDDING_MODEL", "text-embedding-3-large")
EVIDENCE_VECTOR_DIM = int(os.getenv("EVIDENCE_VECTOR_DIM", "3072"))

# Reasoning agents
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "openai")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o")
WORKER_MODEL = os.getenv("WORKER_MODEL", "gpt-4o-mini")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "gpt-4o")
PREPROCESSING_MODEL = os.getenv("PREPROCESSING_MODEL", "gpt-4o-mini")


class NVIDIAAPIError(Exception):
    """Custom exception for NVIDIA API errors."""

    pass


class ImageEncodingError(Exception):
    """Custom exception for image encoding errors."""

    pass


class OpenRouterAPIError(Exception):
    """Custom exception for OpenRouter API errors."""

    pass


class OpenAIAPIError(Exception):
    """Custom exception for OpenAI API errors."""

    pass


class GroqAPIError(Exception):
    """Custom exception for Groq API errors."""

    pass


class GoogleAPIError(Exception):
    """Custom exception for Google API errors."""

    pass


def get_api_key() -> str:
    """Get NVIDIA API key from environment.

    Returns:
        NVIDIA API key

    Raises:
        NVIDIAAPIError: If API key is not found
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise NVIDIAAPIError(
            "NVIDIA_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    return api_key


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment.

    Returns:
        OpenRouter API key

    Raises:
        OpenRouterAPIError: If API key is not found
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterAPIError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    return api_key


def get_groq_api_key() -> str:
    """Get Groq API key from environment.

    Returns:
        Groq API key

    Raises:
        GroqAPIError: If API key is not found
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqAPIError(
            "GROQ_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    return api_key


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment.

    Returns:
        OpenAI API key

    Raises:
        OpenAIAPIError: If API key is not found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIAPIError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    return api_key


def get_google_api_key() -> str:
    """Get Google API key from environment.

    Returns:
        Google API key

    Raises:
        GoogleAPIError: If API key is not found
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GoogleAPIError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it in your .env file or environment."
        )
    return api_key


_EMBEDDING_PROVIDER_BASES = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}

_EMBEDDING_PROVIDER_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
}


# Module-level client caches — avoids re-creating HTTP connection pools per call.
_embedding_client_cache: dict = {}
_reasoning_llm_cache: dict = {}


def get_embedding_client():
    """Return a cached OpenAI-compatible client configured for the active EMBEDDING_PROVIDER.

    Supports: openai (default), openrouter, groq, nvidia.
    Client is cached by provider so the HTTP connection pool is reused.
    """
    from openai import OpenAI

    provider = EMBEDDING_PROVIDER
    if provider in _embedding_client_cache:
        return _embedding_client_cache[provider]

    api_key_env = _EMBEDDING_PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    base_url = _EMBEDDING_PROVIDER_BASES.get(provider)
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    _embedding_client_cache[provider] = client
    return client


def get_reasoning_llm(model: str = None, temperature: float = 0.1, max_tokens: int = 16000):
    """Return a cached LangChain ChatOpenAI instance for the active REASONING_PROVIDER.

    Supports: openai (default), openrouter, groq, nvidia.
    Instance is cached by (provider, model) so the HTTP connection pool is reused.

    Args:
        model: Model name (uses ORCHESTRATOR_MODEL if not provided)
        temperature: LLM temperature
        max_tokens: Maximum tokens for LLM output (default: 16000 for large extractions)

    Returns:
        ChatOpenAI instance configured for the provider
    """
    from langchain_openai import ChatOpenAI

    provider = REASONING_PROVIDER
    model_name = model or ORCHESTRATOR_MODEL
    cache_key = (provider, model_name)

    if cache_key in _reasoning_llm_cache:
        return _reasoning_llm_cache[cache_key]

    api_key_env = _EMBEDDING_PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    base_url = _EMBEDDING_PROVIDER_BASES.get(provider)

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
        "max_tokens": max_tokens,
    }
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOpenAI(**kwargs)
    _reasoning_llm_cache[cache_key] = llm
    return llm


def get_llm_response(prompt: str, provider: str, model: str, temperature: float = 0.3) -> str:
    """Get LLM response for a given prompt using specified provider and model.

    Supports: openai, openrouter, groq, nvidia.

    Args:
        prompt: The prompt to send to the LLM
        provider: LLM provider (openai, groq, nvidia, openrouter)
        model: Model name to use
        temperature: LLM temperature (default: 0.3)

    Returns:
        LLM response text

    Raises:
        Exception: If LLM call fails
    """
    from langchain_openai import ChatOpenAI

    # Get API key based on provider
    api_key_env = _EMBEDDING_PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    
    # Get base URL for non-OpenAI providers
    base_url = _EMBEDDING_PROVIDER_BASES.get(provider)
    
    # Build LLM kwargs
    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    
    # Create LLM and get response
    llm = ChatOpenAI(**kwargs)
    response = llm.invoke(prompt)
    return response.content
