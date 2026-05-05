"""Shared API key management utilities."""

import os


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
