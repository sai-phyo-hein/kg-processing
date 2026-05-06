"""Backward compatibility module - re-exports from api_keys.

This module provides backward compatibility for code that imports from parser.
New code should import directly from api_keys or process_document.document_parser.
"""

# Re-export API exceptions and key getters from api_keys
from kg_extractor.utils.model_setup import (
    NVIDIAAPIError,
    ImageEncodingError,
    OpenRouterAPIError,
    OpenAIAPIError,
    GroqAPIError,
    GoogleAPIError,
    get_api_key,
    get_openrouter_api_key,
    get_groq_api_key,
    get_openai_api_key,
    get_google_api_key,
)

__all__ = [
    "NVIDIAAPIError",
    "ImageEncodingError",
    "OpenRouterAPIError",
    "OpenAIAPIError",
    "GroqAPIError",
    "GoogleAPIError",
    "get_api_key",
    "get_openrouter_api_key",
    "get_groq_api_key",
    "get_openai_api_key",
    "get_google_api_key",
]
