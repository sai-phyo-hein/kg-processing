"""Google provider implementation for document parsing."""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional

import requests

from kg_extractor.utils.model_setup import GoogleAPIError
from ...document_processor import DocumentProcessor, get_parsing_prompt
from ..base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_RETRIES,
    RETRY_DELAY,
)

GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GoogleConfig:
    """Configuration for Google API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stream: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream


def _process_single_page_google(
    page_num: int,
    base64_image: str,
    config: GoogleConfig,
    system_prompt: str,
    content_prompt: str,
) -> Dict[str, Any]:
    """Process a single page using Google API.

    Args:
        page_num: Page number (1-indexed)
        base64_image: Base64 encoded image
        config: Google API configuration
        system_prompt: System prompt for the API
        content_prompt: Content-specific prompt

    Returns:
        Parsed result for the page

    Raises:
        GoogleAPIError: If API call fails
    """
    # Google uses a different format - we need to combine system and content prompts
    combined_prompt = f"{system_prompt}\n\n{content_prompt}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": combined_prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": config.max_tokens,
            "temperature": config.temperature,
            "topP": config.top_p,
        },
    }

    url = f"{GOOGLE_API_URL}/{config.model}:generateContent?key={config.api_key}"

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]

                    try:
                        parsed_content = json.loads(content)
                        parsed_content["page_number"] = page_num
                        return parsed_content
                    except json.JSONDecodeError:
                        return {
                            "page_number": page_num,
                            "content": {"text": content},
                            "metadata": {
                                "has_text": True,
                                "has_diagrams": False,
                                "has_tables": False,
                                "content_quality": "medium",
                            },
                        }
                else:
                    raise GoogleAPIError("Invalid API response format")
            elif response.status_code == 401:
                raise GoogleAPIError("Invalid API key")
            elif response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise GoogleAPIError("Rate limit exceeded")
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]["message"]
                except (ValueError, KeyError):
                    pass
                raise GoogleAPIError(f"API error ({response.status_code}): {error_msg}")

        except requests.exceptions.Timeout:
            last_error = f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES})"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
        except requests.exceptions.RequestException as e:
            last_error = f"Request failed: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue

    raise GoogleAPIError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def process_document_with_google(
    file_path: str,
    config: GoogleConfig,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Process a document using Google API.

    Args:
        file_path: Path to the document file
        config: Google API configuration
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Structured analysis result

    Raises:
        GoogleAPIError: If processing fails
    """
    processor = DocumentProcessor(file_path)
    file_type = processor.get_file_type(file_path)

    if file_type == "image":
        with open(file_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        system_prompt = get_parsing_prompt()
        content_prompt = "Extract all content from this page."

        result = _process_single_page_google(
            1, base64_image, config, system_prompt, content_prompt
        )

        return {
            "file_type": file_type,
            "total_pages": 1,
            "pages": [result],
            "metadata": {
                "processing_model": config.model,
            },
        }

    elif file_type in ["pdf", "docx", "pptx", "xlsx"]:
        pages_data = processor.convert_to_images(pages=pages)

        system_prompt = get_parsing_prompt()
        content_prompt = "Extract all content from this page."

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {
                executor.submit(
                    _process_single_page_google,
                    page_num,
                    base64_image,
                    config,
                    system_prompt,
                    content_prompt,
                ): page_num
                for page_num, base64_image in pages_data.items()
            }

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    results.append({
                        "page_number": page_num,
                        "error": str(e),
                        "content": {"text": f"Error processing page: {e}"},
                    })

        results.sort(key=lambda x: x["page_number"])

        return {
            "file_type": file_type,
            "total_pages": len(pages_data),
            "pages": results,
            "metadata": {
                "processing_model": config.model,
            },
        }

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def extract_text_from_document_google(
    file_path: str,
    config: GoogleConfig,
    pages: Optional[List[int]] = None,
) -> str:
    """Extract text from document using Google API.

    Args:
        file_path: Path to the document file
        config: Google API configuration
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Extracted text content

    Raises:
        GoogleAPIError: If extraction fails
    """
    result = process_document_with_google(file_path, config, pages=pages)

    text_parts = []
    for page in result["pages"]:
        if "content" in page:
            content = page["content"]
            if isinstance(content, dict):
                if "text" in content:
                    text_parts.append(content["text"])
            elif isinstance(content, str):
                text_parts.append(content)

    return "\n\n".join(text_parts)


def extract_text_from_image_google(
    file_path: str,
    config: GoogleConfig,
) -> str:
    """Extract text from image using Google API.

    Args:
        file_path: Path to the image file
        config: Google API configuration

    Returns:
        Extracted text content

    Raises:
        GoogleAPIError: If extraction fails
    """
    with open(file_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    system_prompt = get_parsing_prompt()
    content_prompt = "Extract all content from this page."

    result = _process_single_page_google(
        1, base64_image, config, system_prompt, content_prompt
    )

    if "content" in result:
        content = result["content"]
        if isinstance(content, dict):
            return content.get("text", "")
        elif isinstance(content, str):
            return content

    return ""
