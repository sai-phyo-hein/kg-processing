"""OpenAI provider implementation for document parsing."""
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional

import requests

from kg_extractor.utils.api_keys import OpenAIAPIError
from ...document_processor import DocumentProcessor, get_parsing_prompt
from ..base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_RETRIES,
    RETRY_DELAY,
)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIConfig:
    """Configuration for OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
        temperature: float = 0.20,
        top_p: float = 0.70,
        stream: bool = False,
    ):
        """Initialize OpenAI configuration.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to use streaming response
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream


def _process_single_page_openai(
    page_num: int,
    base64_image: str,
    config: OpenAIConfig,
    system_prompt: str,
    content_prompt: str,
) -> Dict[str, Any]:
    """Process a single page using OpenAI API.

    Args:
        page_num: Page number (1-indexed)
        base64_image: Base64 encoded image
        config: OpenAI API configuration
        system_prompt: System prompt for the API
        content_prompt: Content-specific prompt

    Returns:
        Parsed result for the page

    Raises:
        OpenAIAPIError: If API call fails
    """
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": False,  # Always use non-streaming for structured output
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENAI_API_URL, headers=headers, json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]

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
                    raise OpenAIAPIError("Invalid API response format")

            elif response.status_code == 401:
                raise OpenAIAPIError("Invalid API key")
            elif response.status_code == 429:
                # Rate limit - wait and retry
                wait_time = RETRY_DELAY * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except ValueError:
                    pass
                raise OpenAIAPIError(f"API error ({response.status_code}): {error_msg}")

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

    raise OpenAIAPIError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def process_document_with_openai(
    file_path: str,
    config: OpenAIConfig,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Process a document using OpenAI API.

    Args:
        file_path: Path to the document file
        config: OpenAI API configuration
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Structured analysis result

    Raises:
        OpenAIAPIError: If processing fails
    """
    processor = DocumentProcessor(file_path)
    file_type = processor.get_file_type(file_path)

    if file_type == "image":
        with open(file_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        system_prompt = get_parsing_prompt()
        content_prompt = "Extract all content from this page."

        result = _process_single_page_openai(
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
                    _process_single_page_openai,
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


def extract_text_from_document_openai(
    file_path: str,
    config: OpenAIConfig,
    pages: Optional[List[int]] = None,
) -> str:
    """Extract text from document using OpenAI API.

    Args:
        file_path: Path to the document file
        config: OpenAI API configuration
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Extracted text content

    Raises:
        OpenAIAPIError: If extraction fails
    """
    result = process_document_with_openai(file_path, config, pages=pages)

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


def extract_text_from_image_openai(
    file_path: str,
    config: OpenAIConfig,
) -> str:
    """Extract text from image using OpenAI API.

    Args:
        file_path: Path to the image file
        config: OpenAI API configuration

    Returns:
        Extracted text content

    Raises:
        OpenAIAPIError: If extraction fails
    """
    with open(file_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    system_prompt = get_parsing_prompt()
    content_prompt = "Extract all content from this page."

    result = _process_single_page_openai(
        1, base64_image, config, system_prompt, content_prompt
    )

    if "content" in result:
        content = result["content"]
        if isinstance(content, dict):
            return content.get("text", "")
        elif isinstance(content, str):
            return content

    return ""
