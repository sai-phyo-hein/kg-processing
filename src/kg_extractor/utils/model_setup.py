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

# Reasoning agents
REASONING_PROVIDER = os.getenv("REASONING_PROVIDER", "openai")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o")
WORKER_MODEL = os.getenv("WORKER_MODEL", "gpt-4o-mini")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "gpt-4o")


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


def get_embedding_client():
    """Return an OpenAI-compatible client configured for the active EMBEDDING_PROVIDER.

    Supports: openai (default), openrouter, groq, nvidia.
    """
    from openai import OpenAI

    provider = EMBEDDING_PROVIDER
    api_key_env = _EMBEDDING_PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    base_url = _EMBEDDING_PROVIDER_BASES.get(provider)  # None → default OpenAI URL
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
