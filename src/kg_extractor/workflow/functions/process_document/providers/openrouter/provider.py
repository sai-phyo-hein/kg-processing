"""OpenRouter provider implementation for document parsing."""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional

import requests

from kg_extractor.utils.model_setup import OpenRouterAPIError, ImageEncodingError
from ...document_processor import DocumentProcessor, get_parsing_prompt
from ..base import (
    encode_image,
    extract_text_from_pages,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_RETRIES,
    RETRY_DELAY,
)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterConfig:
    """Configuration for OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemma-3-27b-it",
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


def _process_single_page_openrouter(
    page_num: int,
    base64_image: str,
    config: OpenRouterConfig,
    system_prompt: str,
    content_prompt: str,
) -> Dict[str, Any]:
    """Process a single page using OpenRouter API.

    Args:
        page_num: Page number (1-indexed)
        base64_image: Base64 encoded image
        config: OpenRouter API configuration
        system_prompt: System prompt for the API
        content_prompt: Content-specific prompt

    Returns:
        Parsed result for the page

    Raises:
        OpenRouterAPIError: If API call fails
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
        "HTTP-Referer": "https://github.com/anthropics/claude-code",
        "X-Title": "KG Extractor",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTER_API_URL, headers=headers, json=payload, timeout=60
            )  # Longer timeout for document processing

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]

                    # Handle None content response
                    if content is None:
                        return {
                            "page_number": page_num,
                            "content": {"text": "[Empty page - no content extracted]"},
                            "metadata": {
                                "has_text": False,
                                "has_diagrams": False,
                                "has_tables": False,
                                "content_quality": "low",
                            },
                        }

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
                    raise OpenRouterAPIError("Invalid API response format")
            elif response.status_code == 401:
                raise OpenRouterAPIError("Invalid API key")
            elif response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise OpenRouterAPIError("Rate limit exceeded")
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except ValueError:
                    pass
                raise OpenRouterAPIError(f"API error ({response.status_code}): {error_msg}")

        except requests.exceptions.Timeout:
            last_error = "API request timed out"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

    if last_error:
        raise OpenRouterAPIError(
            f"API call failed for page {page_num} after {MAX_RETRIES} attempts: {last_error}"
        )


def process_document_with_openrouter(
    file_path: str,
    config: OpenRouterConfig,
    pages: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Process document using OpenRouter API with system prompts.

    Args:
        file_path: Path to the document file
        config: OpenRouter API configuration
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Structured analysis result as dictionary

    Raises:
        OpenRouterAPIError: If API call fails
        ImageEncodingError: If image encoding fails
        ValueError: If file format is not supported
    """
    try:
        file_type, base64_images = DocumentProcessor.process_document(file_path, pages=pages)
    except Exception as e:
        raise ValueError(f"Failed to process document: {e}") from e

    system_prompt = get_parsing_prompt()
    content_prompt = "Extract all content from this page."

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        if pages:
            page_numbers = pages
        else:
            page_numbers = list(range(1, len(base64_images) + 1))

        future_to_page = {
            executor.submit(
                _process_single_page_openrouter,
                page_num,
                base64_image,
                config,
                system_prompt,
                content_prompt,
            ): page_num
            for page_num, base64_image in zip(page_numbers, base64_images)
        }

        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise OpenRouterAPIError(f"Failed to process page {page_num}: {e}") from e

    results.sort(key=lambda x: x["page_number"])

    return {
        "file_type": file_type,
        "total_pages": len(results),
        "pages": results,
        "metadata": {
            "processing_model": config.model,
        },
    }


def extract_text_from_document_openrouter(
    file_path: str,
    config: OpenRouterConfig,
    pages: Optional[List[int]] = None,
) -> str:
    """Extract text from document using OpenRouter API.

    Args:
        file_path: Path to the document file
        config: OpenRouter API configuration
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Extracted and structured text content

    Raises:
        OpenRouterAPIError: If API call fails
        ImageEncodingError: If image encoding fails
        ValueError: If file format is not supported
    """
    result = process_document_with_openrouter(file_path, config, pages=pages)
    return extract_text_from_pages(result)


def extract_text_from_image_openrouter(
    image_path: str,
    config: OpenRouterConfig,
    prompt: str = "Extract all text from this image",
) -> str:
    """Send image to OpenRouter API and extract text.

    Args:
        image_path: Path to the image file
        config: OpenRouter API configuration
        prompt: Prompt to send with the image

    Returns:
        Extracted text from the image

    Raises:
        OpenRouterAPIError: If API call fails
        ImageEncodingError: If image encoding fails
    """
    try:
        base64_image = encode_image(image_path)
    except ImageEncodingError as e:
        raise ImageEncodingError(f"Image encoding failed: {e}") from e

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "stream": config.stream,
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anthropics/claude-code",
        "X-Title": "KG Extractor",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    raise OpenRouterAPIError("Invalid API response format")
            elif response.status_code == 401:
                raise OpenRouterAPIError("Invalid API key")
            elif response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise OpenRouterAPIError("Rate limit exceeded")
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except ValueError:
                    pass
                raise OpenRouterAPIError(f"API error ({response.status_code}): {error_msg}")

        except requests.exceptions.Timeout:
            last_error = "API request timed out"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

    raise OpenRouterAPIError(f"API call failed after {MAX_RETRIES} attempts: {last_error}")
