"""Shared utilities for document parsing providers."""
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from kg_extractor.utils.model_setup import ImageEncodingError

# Shared constants
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.20
DEFAULT_TOP_P = 0.70
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_RETRIES = 3
RETRY_DELAY = 1.0


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    path = Path(image_path)
    if not path.exists():
        raise ImageEncodingError(f"Image file not found: {image_path}")
    if not os.access(image_path, os.R_OK):
        raise ImageEncodingError(f"Cannot read image file: {image_path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ImageEncodingError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        raise ImageEncodingError(f"Failed to encode image: {e}") from e


def extract_text_from_pages(result: Dict[str, Any]) -> str:
    """Extract and format text content from page results dict."""
    text_content = []
    for page in result["pages"]:
        if "content" in page:
            if "text" in page["content"]:
                text_content.append(f"--- Page {page['page_number']} ---")
                text_content.append(page["content"]["text"])
            if "diagrams" in page["content"] and page["content"]["diagrams"]:
                text_content.append(f"\n--- Diagrams on Page {page['page_number']} ---")
                for diagram in page["content"]["diagrams"]:
                    text_content.append(f"Type: {diagram.get('type', 'Unknown')}")
                    text_content.append(f"Description: {diagram.get('description', '')}")
                    if "data_insights" in diagram:
                        text_content.append(f"Insights: {diagram['data_insights']}")
            if "tables" in page["content"] and page["content"]["tables"]:
                text_content.append(f"\n--- Tables on Page {page['page_number']} ---")
                for table in page["content"]["tables"]:
                    text_content.append(f"Table: {table.get('title', 'Untitled')}")
                    text_content.append(f"Summary: {table.get('summary', '')}")
                    if "structure" in table:
                        text_content.append(f"Data: {json.dumps(table['structure'], indent=2)}")
    return "\n".join(text_content)
