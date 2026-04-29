"""Parser module for extracting text from images using NVIDIA, OpenRouter, and OpenAI APIs."""

import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import requests
from dotenv import load_dotenv

from kg_extractor.utils.input_processor import (
    DocumentProcessor,
    get_content_specific_prompt,
    get_system_prompt,
)

# Load environment variables
load_dotenv()

# Constants
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "google/gemma-3-27b-it"
DEFAULT_MAX_TOKENS = 2048  # Increased for more detailed responses
DEFAULT_TEMPERATURE = 0.20
DEFAULT_TOP_P = 0.70
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_RETRIES = 3
RETRY_DELAY = 1.0
NVIDIA_RATE_LIMIT_DELAY = 5  # 5 seconds delay between NVIDIA API calls


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


class NVIDIAConfig:
    """Configuration for NVIDIA API calls."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stream: bool = False,
    ) -> None:
        """Initialize NVIDIA API configuration.

        Args:
            api_key: NVIDIA API key for authentication
            model: Model name to use for text extraction
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to use streaming response
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream


class OpenRouterConfig:
    """Configuration for OpenRouter API calls."""

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemma-4-31b-it:free",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stream: bool = False,
    ) -> None:
        """Initialize OpenRouter API configuration.

        Args:
            api_key: OpenRouter API key for authentication
            model: Model name to use for text extraction
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to use streaming response
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream


class GoogleConfig:
    """Configuration for Google API calls."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite-preview",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        stream: bool = False,
    ) -> None:
        """Initialize Google API configuration.

        Args:
            api_key: Google API key for authentication
            model: Model name to use for text extraction
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to use streaming response
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stream = stream


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image

    Raises:
        ImageEncodingError: If image encoding fails
    """
    path = Path(image_path)

    # Validate file exists
    if not path.exists():
        raise ImageEncodingError(f"Image file not found: {image_path}")

    # Validate file is readable
    if not os.access(image_path, os.R_OK):
        raise ImageEncodingError(f"Cannot read image file: {image_path}")

    # Validate file format
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ImageEncodingError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        raise ImageEncodingError(f"Failed to encode image: {e}") from e


def extract_text_from_image(
    image_path: str,
    config: NVIDIAConfig,
    prompt: str = "Extract all text from this image",
) -> str:
    """Send image to NVIDIA API and extract text.

    Args:
        image_path: Path to the image file
        config: NVIDIA API configuration
        prompt: Prompt to send with the image

    Returns:
        Extracted text from the image

    Raises:
        NVIDIAAPIError: If API call fails
        ImageEncodingError: If image encoding fails
    """
    # Encode image
    try:
        base64_image = encode_image(image_path)
    except ImageEncodingError as e:
        raise ImageEncodingError(f"Image encoding failed: {e}") from e

    # Build payload
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

    # Make API call with retry logic
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    # Add delay after successful API call to mitigate rate limiting
                    time.sleep(NVIDIA_RATE_LIMIT_DELAY)
                    return result["choices"][0]["message"]["content"]
                else:
                    raise NVIDIAAPIError("Invalid API response format")
            elif response.status_code == 401:
                raise NVIDIAAPIError("Invalid API key")
            elif response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise NVIDIAAPIError("Rate limit exceeded")
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except ValueError:
                    pass
                raise NVIDIAAPIError(f"API error ({response.status_code}): {error_msg}")

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

    raise NVIDIAAPIError(f"API call failed after {MAX_RETRIES} attempts: {last_error}")


def extract_text_from_image_streaming(
    image_path: str,
    config: NVIDIAConfig,
    prompt: str = "Extract all text from this image",
) -> Iterator[str]:
    """Send image to NVIDIA API with streaming response.

    Args:
        image_path: Path to the image file
        config: NVIDIA API configuration
        prompt: Prompt to send with the image

    Yields:
        Streamed text chunks from the API

    Raises:
        NVIDIAAPIError: If API call fails
        ImageEncodingError: If image encoding fails
    """
    # Encode image
    try:
        base64_image = encode_image(image_path)
    except ImageEncodingError as e:
        raise ImageEncodingError(f"Image encoding failed: {e}") from e

    # Build payload
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
        "stream": True,
    }

    # Make API call
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        response = requests.post(
            NVIDIA_API_URL, headers=headers, json=payload, stream=True, timeout=30
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            import json

                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
            # Add delay after streaming completes to mitigate rate limiting
            time.sleep(NVIDIA_RATE_LIMIT_DELAY)
        elif response.status_code == 401:
            raise NVIDIAAPIError("Invalid API key")
        else:
            error_msg = response.text
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"]
            except ValueError:
                pass
            raise NVIDIAAPIError(f"API error ({response.status_code}): {error_msg}")

    except requests.exceptions.Timeout:
        raise NVIDIAAPIError("API request timed out")
    except requests.exceptions.RequestException as e:
        raise NVIDIAAPIError(f"Network error: {e}")


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


def _process_single_page_nvidia(
    page_num: int,
    base64_image: str,
    config: NVIDIAConfig,
    system_prompt: str,
    content_prompt: str,
) -> Dict[str, Any]:
    """Process a single page using NVIDIA API.

    Args:
        page_num: Page number (1-indexed)
        base64_image: Base64 encoded image
        config: NVIDIA API configuration
        system_prompt: System prompt for the API
        content_prompt: Content-specific prompt

    Returns:
        Parsed result for the page

    Raises:
        NVIDIAAPIError: If API call fails
    """
    # Build payload with system prompt
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
        "top_p": config.top_p,
        "stream": False,  # Always use non-streaming for structured output
    }

    # Make API call
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                NVIDIA_API_URL, headers=headers, json=payload, timeout=60
            )  # Longer timeout for document processing

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]

                    # Try to parse as JSON
                    try:
                        parsed_content = json.loads(content)
                        parsed_content["page_number"] = page_num
                        # Add delay after successful API call to mitigate rate limiting
                        time.sleep(NVIDIA_RATE_LIMIT_DELAY)
                        return parsed_content
                    except json.JSONDecodeError:
                        # If not JSON, wrap in simple structure
                        result_dict = {
                            "page_number": page_num,
                            "content": {"text": content},
                            "metadata": {
                                "has_text": True,
                                "has_diagrams": False,
                                "has_tables": False,
                                "content_quality": "medium",
                            },
                        }
                        # Add delay after successful API call to mitigate rate limiting
                        time.sleep(NVIDIA_RATE_LIMIT_DELAY)
                        return result_dict
                else:
                    raise NVIDIAAPIError("Invalid API response format")
            elif response.status_code == 401:
                raise NVIDIAAPIError("Invalid API key")
            elif response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise NVIDIAAPIError("Rate limit exceeded")
            else:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except ValueError:
                    pass
                raise NVIDIAAPIError(f"API error ({response.status_code}): {error_msg}")

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
        raise NVIDIAAPIError(
            f"API call failed for page {page_num} after {MAX_RETRIES} attempts: {last_error}"
        )


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
    # Build payload with system prompt
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

    # Make API call
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

                    # Try to parse as JSON
                    try:
                        parsed_content = json.loads(content)
                        parsed_content["page_number"] = page_num
                        return parsed_content
                    except json.JSONDecodeError:
                        # If not JSON, wrap in simple structure
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


def process_document_with_api(
    file_path: str,
    config: NVIDIAConfig,
    content_type: str = "mixed",
) -> Dict[str, Any]:
    """Process document using NVIDIA API with system prompts.

    Args:
        file_path: Path to the document file
        config: NVIDIA API configuration
        content_type: Type of content ('text', 'diagram', 'table', 'mixed')

    Returns:
        Structured analysis result as dictionary

    Raises:
        NVIDIAAPIError: If API call fails
        ImageEncodingError: If image encoding fails
        ValueError: If file format is not supported
    """
    # Process document to get images
    try:
        file_type, base64_images = DocumentProcessor.process_document(file_path)
    except Exception as e:
        raise ValueError(f"Failed to process document: {e}") from e

    # Get system and content-specific prompts
    system_prompt = get_system_prompt()
    content_prompt = get_content_specific_prompt(content_type)

    # Process pages in parallel
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all page processing tasks
        future_to_page = {
            executor.submit(
                _process_single_page_nvidia,
                page_num,
                base64_image,
                config,
                system_prompt,
                content_prompt,
            ): page_num
            for page_num, base64_image in enumerate(base64_images, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                results.append(result)
                # Add delay between NVIDIA API calls to mitigate rate limiting
                time.sleep(NVIDIA_RATE_LIMIT_DELAY)
            except Exception as e:
                raise NVIDIAAPIError(f"Failed to process page {page_num}: {e}") from e

    # Sort results by page number to maintain order
    results.sort(key=lambda x: x["page_number"])

    # Combine results
    return {
        "file_type": file_type,
        "total_pages": len(results),
        "pages": results,
        "metadata": {
            "content_type": content_type,
            "processing_model": config.model,
        },
    }


def extract_text_from_document(
    file_path: str,
    config: NVIDIAConfig,
    content_type: str = "mixed",
) -> str:
    """Extract text from document using NVIDIA API.

    Args:
        file_path: Path to the document file
        config: NVIDIA API configuration
        content_type: Type of content ('text', 'diagram', 'table', 'mixed')

    Returns:
        Extracted and structured text content

    Raises:
        NVIDIAAPIError: If API call fails
        ImageEncodingError: If image encoding fails
        ValueError: If file format is not supported
    """
    result = process_document_with_api(file_path, config, content_type)

    # Extract and combine text from all pages
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


def process_document_with_openrouter(
    file_path: str,
    config: OpenRouterConfig,
    content_type: str = "mixed",
) -> Dict[str, Any]:
    """Process document using OpenRouter API with system prompts.

    Args:
        file_path: Path to the document file
        config: OpenRouter API configuration
        content_type: Type of content ('text', 'diagram', 'table', 'mixed')

    Returns:
        Structured analysis result as dictionary

    Raises:
        OpenRouterAPIError: If API call fails
        ImageEncodingError: If image encoding fails
        ValueError: If file format is not supported
    """
    # Process document to get images
    try:
        file_type, base64_images = DocumentProcessor.process_document(file_path)
    except Exception as e:
        raise ValueError(f"Failed to process document: {e}") from e

    # Get system and content-specific prompts
    system_prompt = get_system_prompt()
    content_prompt = get_content_specific_prompt(content_type)

    # Process pages in parallel
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all page processing tasks
        future_to_page = {
            executor.submit(
                _process_single_page_openrouter,
                page_num,
                base64_image,
                config,
                system_prompt,
                content_prompt,
            ): page_num
            for page_num, base64_image in enumerate(base64_images, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                raise OpenRouterAPIError(f"Failed to process page {page_num}: {e}") from e

    # Sort results by page number to maintain order
    results.sort(key=lambda x: x["page_number"])

    # Combine results
    return {
        "file_type": file_type,
        "total_pages": len(results),
        "pages": results,
        "metadata": {
            "content_type": content_type,
            "processing_model": config.model,
        },
    }


def extract_text_from_document_openrouter(
    file_path: str,
    config: OpenRouterConfig,
    content_type: str = "mixed",
) -> str:
    """Extract text from document using OpenRouter API.

    Args:
        file_path: Path to the document file
        config: OpenRouter API configuration
        content_type: Type of content ('text', 'diagram', 'table', 'mixed')

    Returns:
        Extracted and structured text content

    Raises:
        OpenRouterAPIError: If API call fails
        ImageEncodingError: If image encoding fails
        ValueError: If file format is not supported
    """
    result = process_document_with_openrouter(file_path, config, content_type)

    # Extract and combine text from all pages
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
    # Encode image
    try:
        base64_image = encode_image(image_path)
    except ImageEncodingError as e:
        raise ImageEncodingError(f"Image encoding failed: {e}") from e

    # Build payload
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

    # Make API call with retry logic
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


def extract_text_from_image_streaming_openrouter(
    image_path: str,
    config: OpenRouterConfig,
    prompt: str = "Extract all text from this image",
) -> Iterator[str]:
    """Send image to OpenRouter API with streaming response.

    Args:
        image_path: Path to the image file
        config: OpenRouter API configuration
        prompt: Prompt to send with the image

    Yields:
        Streamed text chunks from the API

    Raises:
        OpenRouterAPIError: If API call fails
        ImageEncodingError: If image encoding fails
    """
    # Encode image
    try:
        base64_image = encode_image(image_path)
    except ImageEncodingError as e:
        raise ImageEncodingError(f"Image encoding failed: {e}") from e

    # Build payload
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
        "stream": True,
    }

    # Make API call
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anthropics/claude-code",
        "X-Title": "KG Extractor",
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL, headers=headers, json=payload, stream=True, timeout=30
        )

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
        elif response.status_code == 401:
            raise OpenRouterAPIError("Invalid API key")
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
        raise OpenRouterAPIError("Request timeout")
    except requests.exceptions.RequestException as e:
        raise OpenRouterAPIError(f"Request failed: {e}")


# OpenAI-specific functions

def _process_single_page_openai(
    page_num: int,
    base64_image: str,
    config: "OpenAIConfig",
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
    # Build payload with system prompt
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

    # Make API call
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

                    # Try to parse as JSON
                    try:
                        parsed_content = json.loads(content)
                        parsed_content["page_number"] = page_num
                        return parsed_content
                    except json.JSONDecodeError:
                        # If not JSON, wrap in simple structure
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
    config: "OpenAIConfig",
    content_type: str = "mixed",
) -> Dict[str, Any]:
    """Process a document using OpenAI API.

    Args:
        file_path: Path to the document file
        config: OpenAI API configuration
        content_type: Type of content to focus on

    Returns:
        Structured analysis result

    Raises:
        OpenAIAPIError: If processing fails
    """
    processor = DocumentProcessor(file_path)
    file_type = processor.get_file_type(file_path)

    if file_type == "image":
        # Handle single image
        with open(file_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        system_prompt = get_system_prompt()
        content_prompt = get_content_specific_prompt(content_type)

        result = _process_single_page_openai(
            1, base64_image, config, system_prompt, content_prompt
        )

        return {
            "file_type": file_type,
            "total_pages": 1,
            "pages": [result],
            "metadata": {
                "processing_model": config.model,
                "content_type": content_type,
            },
        }

    elif file_type in ["pdf", "docx", "pptx", "xlsx"]:
        # Handle multi-page documents
        pages_data = processor.convert_to_images()

        system_prompt = get_system_prompt()
        content_prompt = get_content_specific_prompt(content_type)

        # Process pages in parallel
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

        # Sort results by page number
        results.sort(key=lambda x: x["page_number"])

        return {
            "file_type": file_type,
            "total_pages": len(pages_data),
            "pages": results,
            "metadata": {
                "processing_model": config.model,
                "content_type": content_type,
            },
        }

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def extract_text_from_document_openai(
    file_path: str,
    config: "OpenAIConfig",
    content_type: str = "mixed",
) -> str:
    """Extract text from document using OpenAI API.

    Args:
        file_path: Path to the document file
        config: OpenAI API configuration
        content_type: Type of content to focus on

    Returns:
        Extracted text content

    Raises:
        OpenAIAPIError: If extraction fails
    """
    result = process_document_with_openai(file_path, config, content_type)

    # Extract text from all pages
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
    config: "OpenAIConfig",
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

    system_prompt = get_system_prompt()
    content_prompt = get_content_specific_prompt("mixed")

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


def extract_text_from_image_streaming_openai(
    file_path: str,
    config: "OpenAIConfig",
) -> Iterator[str]:
    """Extract text from image using OpenAI API with streaming.

    Args:
        file_path: Path to the image file
        config: OpenAI API configuration

    Yields:
        Text chunks as they are received

    Raises:
        OpenAIAPIError: If extraction fails
    """
    with open(file_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    system_prompt = get_system_prompt()
    content_prompt = get_content_specific_prompt("mixed")

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
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            OPENAI_API_URL, headers=headers, json=payload, stream=True, timeout=60
        )

        if response.status_code == 401:
            raise OpenAIAPIError("Invalid API key")
        elif response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"]
            except ValueError:
                pass
            raise OpenAIAPIError(f"API error ({response.status_code}): {error_msg}")

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    except requests.exceptions.Timeout:
        raise OpenAIAPIError("API request timed out")
    except requests.exceptions.RequestException as e:
        raise OpenAIAPIError(f"Network error: {e}")


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


# Google-specific functions

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
    # Build payload for Google Gemini API
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

    # Make API call
    url = f"{GOOGLE_API_URL}/{config.model}:generateContent?key={config.api_key}"

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]

                    # Try to parse as JSON
                    try:
                        parsed_content = json.loads(content)
                        parsed_content["page_number"] = page_num
                        return parsed_content
                    except json.JSONDecodeError:
                        # If not JSON, wrap in simple structure
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
    content_type: str = "mixed",
) -> Dict[str, Any]:
    """Process a document using Google API.

    Args:
        file_path: Path to the document file
        config: Google API configuration
        content_type: Type of content to focus on

    Returns:
        Structured analysis result

    Raises:
        GoogleAPIError: If processing fails
    """
    processor = DocumentProcessor(file_path)
    file_type = processor.get_file_type(file_path)

    if file_type == "image":
        # Handle single image
        with open(file_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        system_prompt = get_system_prompt()
        content_prompt = get_content_specific_prompt(content_type)

        result = _process_single_page_google(
            1, base64_image, config, system_prompt, content_prompt
        )

        return {
            "file_type": file_type,
            "total_pages": 1,
            "pages": [result],
            "metadata": {
                "processing_model": config.model,
                "content_type": content_type,
            },
        }

    elif file_type in ["pdf", "docx", "pptx", "xlsx"]:
        # Handle multi-page documents
        pages_data = processor.convert_to_images()

        system_prompt = get_system_prompt()
        content_prompt = get_content_specific_prompt(content_type)

        # Process pages in parallel
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

        # Sort results by page number
        results.sort(key=lambda x: x["page_number"])

        return {
            "file_type": file_type,
            "total_pages": len(pages_data),
            "pages": results,
            "metadata": {
                "processing_model": config.model,
                "content_type": content_type,
            },
        }

    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def extract_text_from_document_google(
    file_path: str,
    config: GoogleConfig,
    content_type: str = "mixed",
) -> str:
    """Extract text from document using Google API.

    Args:
        file_path: Path to the document file
        config: Google API configuration
        content_type: Type of content to focus on

    Returns:
        Extracted text content

    Raises:
        GoogleAPIError: If extraction fails
    """
    result = process_document_with_google(file_path, config, content_type)

    # Extract text from all pages
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

    system_prompt = get_system_prompt()
    content_prompt = get_content_specific_prompt("mixed")

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


def extract_text_from_image_streaming_google(
    file_path: str,
    config: GoogleConfig,
) -> Iterator[str]:
    """Extract text from image using Google API with streaming.

    Args:
        file_path: Path to the image file
        config: Google API configuration

    Yields:
        Text chunks as they are received

    Raises:
        GoogleAPIError: If extraction fails
    """
    with open(file_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")

    system_prompt = get_system_prompt()
    content_prompt = get_content_specific_prompt("mixed")
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
        "stream": True,
    }

    url = f"{GOOGLE_API_URL}/{config.model}:streamGenerateContent?key={config.api_key}"

    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)

        if response.status_code == 401:
            raise GoogleAPIError("Invalid API key")
        elif response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"]["message"]
            except (ValueError, KeyError):
                pass
            raise GoogleAPIError(f"API error ({response.status_code}): {error_msg}")

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    try:
                        data = json.loads(data_str)
                        if "candidates" in data and len(data["candidates"]) > 0:
                            content = data["candidates"][0].get("content", {})
                            parts = content.get("parts", [])
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
                    except json.JSONDecodeError:
                        continue

    except requests.exceptions.Timeout:
        raise GoogleAPIError("API request timed out")
    except requests.exceptions.RequestException as e:
        raise GoogleAPIError(f"Network error: {e}")
