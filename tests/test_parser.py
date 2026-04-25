"""Tests for kg-extractor parser module."""

import base64
import os
from unittest.mock import Mock, patch

import pytest
import requests

from kg_extractor.parser import (
    SUPPORTED_IMAGE_FORMATS,
    ImageEncodingError,
    NVIDIAAPIError,
    NVIDIAConfig,
    encode_image,
    extract_text_from_image,
    extract_text_from_image_streaming,
    get_api_key,
)


class TestNVIDIAConfig:
    """Tests for NVIDIAConfig class."""

    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = NVIDIAConfig(
            api_key="test_key",
            model="google/gemma-3-27b-it",
            max_tokens=512,
            temperature=0.20,
            top_p=0.70,
            stream=False,
        )
        assert config.api_key == "test_key"
        assert config.model == "google/gemma-3-27b-it"
        assert config.max_tokens == 512
        assert config.temperature == 0.20
        assert config.top_p == 0.70
        assert config.stream is False

    def test_empty_api_key_raises_error(self):
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            NVIDIAConfig(api_key="")

    def test_invalid_max_tokens_raises_error(self):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            NVIDIAConfig(api_key="test_key", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            NVIDIAConfig(api_key="test_key", max_tokens=-100)

    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            NVIDIAConfig(api_key="test_key", temperature=-0.1)

        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            NVIDIAConfig(api_key="test_key", temperature=2.1)

    def test_invalid_top_p_raises_error(self):
        """Test that invalid top_p raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            NVIDIAConfig(api_key="test_key", top_p=-0.1)

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            NVIDIAConfig(api_key="test_key", top_p=1.1)


class TestEncodeImage:
    """Tests for encode_image function."""

    def test_encode_valid_image(self, tmp_path):
        """Test encoding a valid image file."""
        # Create a test image file
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")

        result = encode_image(str(test_image))
        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == b"fake_image_data"

    def test_encode_nonexistent_file(self):
        """Test encoding a non-existent file raises error."""
        with pytest.raises(ImageEncodingError, match="Image file not found"):
            encode_image("/nonexistent/file.png")

    def test_encode_unreadable_file(self, tmp_path):
        """Test encoding an unreadable file raises error."""
        # Create a file and make it unreadable
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")
        os.chmod(test_image, 0o000)

        with pytest.raises(ImageEncodingError, match="Cannot read image file"):
            encode_image(str(test_image))

        # Clean up
        os.chmod(test_image, 0o644)

    def test_encode_unsupported_format(self, tmp_path):
        """Test encoding unsupported format raises error."""
        # Create a file with unsupported extension
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"some data")

        with pytest.raises(ImageEncodingError, match="Unsupported image format"):
            encode_image(str(test_file))

    def test_supported_formats(self):
        """Test that supported formats are correct."""
        assert ".png" in SUPPORTED_IMAGE_FORMATS
        assert ".jpg" in SUPPORTED_IMAGE_FORMATS
        assert ".jpeg" in SUPPORTED_IMAGE_FORMATS
        assert ".webp" in SUPPORTED_IMAGE_FORMATS


class TestExtractTextFromImage:
    """Tests for extract_text_from_image function."""

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_successful_extraction(self, mock_encode, mock_post):
        """Test successful text extraction."""
        # Setup mocks
        mock_encode.return_value = "base64_encoded_image"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Extracted text"}}]}
        mock_post.return_value = mock_response

        config = NVIDIAConfig(api_key="test_key")
        result = extract_text_from_image("test.jpg", config)

        assert result == "Extracted text"
        mock_post.assert_called_once()

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_invalid_api_key(self, mock_encode, mock_post):
        """Test handling of invalid API key."""
        mock_encode.return_value = "base64_encoded_image"
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        config = NVIDIAConfig(api_key="invalid_key")
        with pytest.raises(NVIDIAAPIError, match="Invalid API key"):
            extract_text_from_image("test.jpg", config)

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_rate_limit_with_retry(self, mock_encode, mock_post):
        """Test handling of rate limit with retry."""
        mock_encode.return_value = "base64_encoded_image"

        # First call fails with rate limit, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 429

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Extracted text"}}]
        }

        mock_post.side_effect = [mock_response_fail, mock_response_success]

        config = NVIDIAConfig(api_key="test_key")
        result = extract_text_from_image("test.jpg", config)

        assert result == "Extracted text"
        assert mock_post.call_count == 2

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_timeout_error(self, mock_encode, mock_post):
        """Test handling of timeout error."""
        mock_encode.return_value = "base64_encoded_image"
        mock_post.side_effect = requests.exceptions.Timeout()

        config = NVIDIAConfig(api_key="test_key")
        with pytest.raises(NVIDIAAPIError, match="API call failed"):
            extract_text_from_image("test.jpg", config)

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_network_error(self, mock_encode, mock_post):
        """Test handling of network error."""
        mock_encode.return_value = "base64_encoded_image"
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        config = NVIDIAConfig(api_key="test_key")
        with pytest.raises(NVIDIAAPIError, match="API call failed"):
            extract_text_from_image("test.jpg", config)

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_invalid_response_format(self, mock_encode, mock_post):
        """Test handling of invalid response format."""
        mock_encode.return_value = "base64_encoded_image"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response"}
        mock_post.return_value = mock_response

        config = NVIDIAConfig(api_key="test_key")
        with pytest.raises(NVIDIAAPIError, match="Invalid API response format"):
            extract_text_from_image("test.jpg", config)


class TestExtractTextFromImageStreaming:
    """Tests for extract_text_from_image_streaming function."""

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_successful_streaming(self, mock_encode, mock_post):
        """Test successful streaming text extraction."""
        mock_encode.return_value = "base64_encoded_image"

        # Create mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200

        # Simulate streaming lines
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b"data: [DONE]",
        ]
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        config = NVIDIAConfig(api_key="test_key")
        result = list(extract_text_from_image_streaming("test.jpg", config))

        assert result == ["Hello", " world"]

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_streaming_invalid_api_key(self, mock_encode, mock_post):
        """Test streaming with invalid API key."""
        mock_encode.return_value = "base64_encoded_image"
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        config = NVIDIAConfig(api_key="invalid_key")
        with pytest.raises(NVIDIAAPIError, match="Invalid API key"):
            list(extract_text_from_image_streaming("test.jpg", config))

    @patch("kg_extractor.parser.requests.post")
    @patch("kg_extractor.parser.encode_image")
    def test_streaming_timeout(self, mock_encode, mock_post):
        """Test streaming with timeout."""
        mock_encode.return_value = "base64_encoded_image"
        mock_post.side_effect = requests.exceptions.Timeout()

        config = NVIDIAConfig(api_key="test_key")
        with pytest.raises(NVIDIAAPIError, match="API request timed out"):
            list(extract_text_from_image_streaming("test.jpg", config))


class TestGetAPIKey:
    """Tests for get_api_key function."""

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    def test_get_api_key_from_env(self):
        """Test getting API key from environment variable."""
        result = get_api_key()
        assert result == "test_key"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_missing(self):
        """Test error when API key is missing."""
        with pytest.raises(NVIDIAAPIError, match="NVIDIA_API_KEY environment variable not set"):
            get_api_key()


class TestIntegration:
    """Integration tests for the parser module."""

    def test_end_to_end_flow_with_mock(self, tmp_path):
        """Test end-to-end flow with mocked API."""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")

        with patch("kg_extractor.parser.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "Test result"}}]}
            mock_post.return_value = mock_response

            config = NVIDIAConfig(api_key="test_key")
            result = extract_text_from_image(str(test_image), config)

            assert result == "Test result"

    def test_error_handling_flow(self, tmp_path):
        """Test error handling in the flow."""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")

        with patch("kg_extractor.parser.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("Network error")

            config = NVIDIAConfig(api_key="test_key")
            with pytest.raises(NVIDIAAPIError):
                extract_text_from_image(str(test_image), config)
