"""Process document functions."""

from .format_markdown import save_markdown_result, save_text_markdown
from .process_docx import process_docx
from .process_image import process_image
from .process_pdf import process_pdf
from .process_pptx import process_pptx
from .process_xlsx import process_xlsx
from .document_processor import DocumentProcessor, get_parsing_prompt
from .document_parser import (
    NVIDIAConfig,
    OpenRouterConfig,
    OpenAIConfig,
    GoogleConfig,
    extract_text_from_document,
    extract_text_from_document_openrouter,
    extract_text_from_document_openai,
    extract_text_from_document_google,
    process_document_with_nvidia,
    process_document_with_openrouter,
    process_document_with_openai,
    process_document_with_google,
)
# Re-export API key functions for convenience
from kg_extractor.utils.model_setup import (
    get_api_key,
    get_openrouter_api_key,
    get_openai_api_key,
    get_google_api_key,
)

__all__ = [
    "save_markdown_result",
    "save_text_markdown",
    "process_docx",
    "process_image",
    "process_pdf",
    "process_pptx",
    "process_xlsx",
    "DocumentProcessor",
    "get_parsing_prompt",
    "NVIDIAConfig",
    "OpenRouterConfig",
    "OpenAIConfig",
    "GoogleConfig",
    "extract_text_from_document",
    "extract_text_from_document_openrouter",
    "extract_text_from_document_openai",
    "extract_text_from_document_google",
    "process_document_with_nvidia",
    "process_document_with_openrouter",
    "process_document_with_openai",
    "process_document_with_google",
    "get_api_key",
    "get_openrouter_api_key",
    "get_openai_api_key",
    "get_google_api_key",
]

