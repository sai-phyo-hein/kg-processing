"""Tests for kg-extractor main module."""

from unittest.mock import patch

import pytest

from kg_extractor import __version__
from kg_extractor.main import main


class TestVersion:
    """Tests for version information."""

    def test_version_is_defined(self):
        """Test that version is defined."""
        assert __version__ == "0.1.0"
        assert isinstance(__version__, str)


class TestMainCLI:
    """Tests for main CLI functionality."""

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg"])
    def test_main_with_valid_image(self, mock_extract, mock_get_key, tmp_path):
        """Test main with valid image."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.return_value = "Extracted text"

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image)]):
            with patch("builtins.print") as mock_print:
                try:
                    main()
                except SystemExit:
                    pass

                # Check that extracted text was printed
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert "Extracted text" in print_calls

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg", "--output", "result.txt"])
    def test_main_with_output_file(self, mock_extract, mock_get_key, tmp_path):
        """Test main with output file."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.return_value = "Extracted text"

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        output_file = tmp_path / "result.txt"

        with patch("sys.argv", ["kg-extractor", str(test_image), "--output", str(output_file)]):
            try:
                main()
            except SystemExit:
                pass

            # Check that output file was created
            assert output_file.exists()
            content = output_file.read_text()
            assert content == "Extracted text"

    @patch("sys.argv", ["kg-extractor", "--help"])
    def test_main_help(self):
        """Test main help message."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("sys.argv", ["kg-extractor", "--version"])
    def test_main_version(self):
        """Test main version flag."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("sys.argv", ["kg-extractor", "nonexistent.jpg"])
    def test_main_missing_file(self):
        """Test main with missing file."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("kg_extractor.main.get_api_key")
    @patch("sys.argv", ["kg-extractor", "test.jpg"])
    def test_main_missing_api_key(self, mock_get_key, tmp_path):
        """Test main with missing API key."""
        # Setup mock to raise error
        from kg_extractor.parser import NVIDIAAPIError

        mock_get_key.side_effect = NVIDIAAPIError("NVIDIA_API_KEY environment variable not set")

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.type is SystemExit

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image_streaming")
    @patch("sys.argv", ["kg-extractor", "test.jpg", "--stream"])
    def test_main_streaming_mode(self, mock_extract, mock_get_key, tmp_path):
        """Test main with streaming mode."""
        # Setup mocks
        mock_get_key.return_value = "test_key"

        # Mock streaming generator
        def mock_streaming(*args, **kwargs):
            yield "Hello"
            yield " world"

        mock_extract.side_effect = mock_streaming

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image), "--stream"]):
            with patch("builtins.print") as mock_print:
                try:
                    main()
                except SystemExit:
                    pass

                # Check that streaming chunks were printed
                # The print function is called with end="" so we need to check the calls
                assert mock_print.call_count >= 2
                # Check that "Hello" and " world" were printed
                all_printed = "".join(
                    [str(call[0][0]) if call[0] else "" for call in mock_print.call_args_list]
                )
                assert "Hello" in all_printed
                assert " world" in all_printed

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg", "--model", "custom-model"])
    def test_main_custom_model(self, mock_extract, mock_get_key, tmp_path):
        """Test main with custom model."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.return_value = "Result"

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image), "--model", "custom-model"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify extract was called with correct config
            mock_extract.assert_called_once()
            call_args = mock_extract.call_args
            assert call_args[0][0] == str(test_image)
            assert call_args[0][1].model == "custom-model"

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg", "--max-tokens", "1024"])
    def test_main_custom_max_tokens(self, mock_extract, mock_get_key, tmp_path):
        """Test main with custom max_tokens."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.return_value = "Result"

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image), "--max-tokens", "1024"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify extract was called with correct config
            call_args = mock_extract.call_args
            assert call_args[0][1].max_tokens == 1024

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg", "--temperature", "0.5"])
    def test_main_custom_temperature(self, mock_extract, mock_get_key, tmp_path):
        """Test main with custom temperature."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.return_value = "Result"

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image), "--temperature", "0.5"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify extract was called with correct config
            call_args = mock_extract.call_args
            assert call_args[0][1].temperature == 0.5

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg", "--top-p", "0.9"])
    def test_main_custom_top_p(self, mock_extract, mock_get_key, tmp_path):
        """Test main with custom top_p."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.return_value = "Result"

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image), "--top-p", "0.9"]):
            try:
                main()
            except SystemExit:
                pass

            # Verify extract was called with correct config
            call_args = mock_extract.call_args
            assert call_args[0][1].top_p == 0.9

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg"])
    def test_main_keyboard_interrupt(self, mock_extract, mock_get_key, tmp_path):
        """Test main handles keyboard interrupt."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.side_effect = KeyboardInterrupt()

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 130

    @patch("kg_extractor.main.get_api_key")
    @patch("kg_extractor.main.extract_text_from_image")
    @patch("sys.argv", ["kg-extractor", "test.jpg"])
    def test_main_unexpected_error(self, mock_extract, mock_get_key, tmp_path):
        """Test main handles unexpected errors."""
        # Setup mocks
        mock_get_key.return_value = "test_key"
        mock_extract.side_effect = Exception("Unexpected error")

        # Create test image
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b"fake_image")

        with patch("sys.argv", ["kg-extractor", str(test_image)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
