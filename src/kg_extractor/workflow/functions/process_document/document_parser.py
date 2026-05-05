"""Document parser - backward compatibility re-exports."""
from .providers.base import encode_image, extract_text_from_pages
from .providers.nvidia.provider import (
    NVIDIAConfig,
    extract_text_from_image,
    process_document_with_nvidia,
    extract_text_from_document,
)
from .providers.openrouter.provider import (
    OpenRouterConfig,
    process_document_with_openrouter,
    extract_text_from_document_openrouter,
    extract_text_from_image_openrouter,
)
from .providers.openai.provider import (
    OpenAIConfig,
    process_document_with_openai,
    extract_text_from_document_openai,
    extract_text_from_image_openai,
)
from .providers.google.provider import (
    GoogleConfig,
    process_document_with_google,
    extract_text_from_document_google,
    extract_text_from_image_google,
)

__all__ = [
    "encode_image", "extract_text_from_pages",
    "NVIDIAConfig", "extract_text_from_image",
    "process_document_with_nvidia", "extract_text_from_document",
    "OpenRouterConfig", "process_document_with_openrouter", "extract_text_from_document_openrouter",
    "extract_text_from_image_openrouter",
    "OpenAIConfig", "process_document_with_openai", "extract_text_from_document_openai",
    "extract_text_from_image_openai",
    "GoogleConfig", "process_document_with_google", "extract_text_from_document_google",
    "extract_text_from_image_google",
]
