"""Provider implementations for document parsing."""
from .nvidia.provider import (
    NVIDIAConfig,
    extract_text_from_image,
    process_document_with_nvidia,
    extract_text_from_document,
)
from .openrouter.provider import (
    OpenRouterConfig,
    process_document_with_openrouter,
    extract_text_from_document_openrouter,
    extract_text_from_image_openrouter,
)
from .openai.provider import (
    OpenAIConfig,
    process_document_with_openai,
    extract_text_from_document_openai,
    extract_text_from_image_openai,
)
from .google.provider import (
    GoogleConfig,
    process_document_with_google,
    extract_text_from_document_google,
    extract_text_from_image_google,
)
from .base import encode_image, extract_text_from_pages
