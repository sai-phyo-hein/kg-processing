"""Tests for kg-extractor input processor module."""

import base64
from unittest.mock import MagicMock, Mock, patch

import pytest

from kg_extractor.docx_processor import process_docx
from kg_extractor.image_processor import process_image
from kg_extractor.input_processor import (
    DocumentProcessor,
    get_content_specific_prompt,
    get_system_prompt,
)
from kg_extractor.pdf_processor import process_pdf
from kg_extractor.pptx_processor import process_pptx
from kg_extractor.xlsx_processor import process_xlsx


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    def test_get_file_type_image(self):
        """Test file type detection for images."""
        assert DocumentProcessor.get_file_type("test.png") == "image"
        assert DocumentProcessor.get_file_type("test.jpg") == "image"
        assert DocumentProcessor.get_file_type("test.jpeg") == "image"
        assert DocumentProcessor.get_file_type("test.webp") == "image"

    def test_get_file_type_pdf(self):
        """Test file type detection for PDF."""
        assert DocumentProcessor.get_file_type("test.pdf") == "pdf"

    def test_get_file_type_docx(self):
        """Test file type detection for DOCX."""
        assert DocumentProcessor.get_file_type("test.docx") == "docx"
        assert DocumentProcessor.get_file_type("test.doc") == "docx"

    def test_get_file_type_pptx(self):
        """Test file type detection for PPTX."""
        assert DocumentProcessor.get_file_type("test.pptx") == "pptx"
        assert DocumentProcessor.get_file_type("test.ppt") == "pptx"

    def test_get_file_type_xlsx(self):
        """Test file type detection for XLSX."""
        assert DocumentProcessor.get_file_type("test.xlsx") == "xlsx"
        assert DocumentProcessor.get_file_type("test.xls") == "xlsx"

    def test_get_file_type_unknown(self):
        """Test file type detection for unknown formats."""
        assert DocumentProcessor.get_file_type("test.txt") == "unknown"
        assert DocumentProcessor.get_file_type("test.unknown") == "unknown"

    def test_process_image(self, tmp_path):
        """Test processing an image file."""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")

        result = process_image(str(test_image))
        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == b"fake_image_data"

    @patch("kg_extractor.pdf_processor.fitz")
    @patch("kg_extractor.pdf_processor.Image")
    def test_process_pdf(self, mock_image, mock_fitz, tmp_path):
        """Test processing a PDF file."""
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake_pdf_data")

        # Mock PyMuPDF
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=1)
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.width = 100
        mock_pix.height = 100
        mock_pix.samples = b"fake_image_samples"
        mock_page.get_pixmap = Mock(return_value=mock_pix)
        mock_doc.__getitem__ = Mock(return_value=mock_page)
        mock_fitz.open = Mock(return_value=mock_doc)

        # Mock PIL Image
        mock_img = MagicMock()
        mock_image.frombytes = Mock(return_value=mock_img)
        mock_img.save = Mock()

        result = process_pdf(str(test_pdf))
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(img, str) for img in result)

    @patch("kg_extractor.docx_processor.Document")
    def test_process_docx(self, mock_document, tmp_path):
        """Test processing a DOCX file."""
        # Create a test DOCX file
        test_docx = tmp_path / "test.docx"
        test_docx.write_bytes(b"fake_docx_data")

        # Mock python-docx
        mock_doc = MagicMock()
        mock_para = MagicMock()
        mock_para.text = "Test paragraph"
        mock_doc.paragraphs = [mock_para]
        mock_document.return_value = mock_doc

        result = process_docx(str(test_docx))
        assert isinstance(result, list)
        assert len(result) >= 1

    @patch("kg_extractor.pptx_processor.Presentation")
    def test_process_pptx(self, mock_presentation, tmp_path):
        """Test processing a PPTX file."""
        # Create a test PPTX file
        test_pptx = tmp_path / "test.pptx"
        test_pptx.write_bytes(b"fake_pptx_data")

        # Mock python-pptx
        mock_prs = MagicMock()
        mock_slide = MagicMock()
        mock_shape = MagicMock()
        mock_shape.text = "Test slide text"
        mock_slide.shapes = [mock_shape]
        mock_prs.slides = [mock_slide]
        mock_presentation.return_value = mock_prs

        result = process_pptx(str(test_pptx))
        assert isinstance(result, list)
        assert len(result) >= 1

    @patch("kg_extractor.xlsx_processor.load_workbook")
    def test_process_xlsx(self, mock_workbook, tmp_path):
        """Test processing an Excel file."""
        # Create a test Excel file
        test_xlsx = tmp_path / "test.xlsx"
        test_xlsx.write_bytes(b"fake_xlsx_data")

        # Mock openpyxl
        mock_wb = MagicMock()
        mock_sheet = MagicMock()
        mock_sheet.iter_rows = Mock(return_value=[["Data1", "Data2"]])
        mock_wb.worksheets = [mock_sheet]
        mock_workbook.return_value = mock_wb

        result = process_xlsx(str(test_xlsx))
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_process_document_image(self, tmp_path):
        """Test processing an image document."""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")

        file_type, images = DocumentProcessor.process_document(str(test_image))
        assert file_type == "image"
        assert len(images) == 1
        assert isinstance(images[0], str)

    def test_process_document_unsupported(self, tmp_path):
        """Test processing an unsupported document format."""
        # Create a test file with unsupported extension
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"some data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            DocumentProcessor.process_document(str(test_file))


class TestSystemPrompts:
    """Tests for system prompt functions."""

    def test_get_system_prompt(self):
        """Test getting system prompt."""
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "text" in prompt.lower()
        assert "diagram" in prompt.lower()
        assert "table" in prompt.lower()
        assert "json" in prompt.lower()

    def test_get_content_specific_prompt_text(self):
        """Test getting text-specific prompt."""
        prompt = get_content_specific_prompt("text")
        assert isinstance(prompt, str)
        assert "text" in prompt.lower()

    def test_get_content_specific_prompt_diagram(self):
        """Test getting diagram-specific prompt."""
        prompt = get_content_specific_prompt("diagram")
        assert isinstance(prompt, str)
        assert "diagram" in prompt.lower()

    def test_get_content_specific_prompt_table(self):
        """Test getting table-specific prompt."""
        prompt = get_content_specific_prompt("table")
        assert isinstance(prompt, str)
        assert "table" in prompt.lower()

    def test_get_content_specific_prompt_mixed(self):
        """Test getting mixed content prompt."""
        prompt = get_content_specific_prompt("mixed")
        assert isinstance(prompt, str)
        # Should contain references to all content types
        assert "text" in prompt.lower() or "diagram" in prompt.lower() or "table" in prompt.lower()

    def test_get_content_specific_prompt_unknown(self):
        """Test getting prompt for unknown content type."""
        prompt = get_content_specific_prompt("unknown")
        assert isinstance(prompt, str)
        # Should default to mixed, so should contain some content type references
        assert len(prompt) > 0


class TestIntegration:
    """Integration tests for input processor."""

    def test_end_to_end_image_processing(self, tmp_path):
        """Test end-to-end image processing."""
        # Create a test image
        test_image = tmp_path / "test.png"
        test_image.write_bytes(b"fake_image_data")

        file_type, images = DocumentProcessor.process_document(str(test_image))
        assert file_type == "image"
        assert len(images) == 1

        # Verify base64 encoding
        decoded = base64.b64decode(images[0])
        assert decoded == b"fake_image_data"

    def test_system_prompt_completeness(self):
        """Test that system prompt contains all required elements."""
        prompt = get_system_prompt()

        # Check for key concepts (case-insensitive)
        required_concepts = [
            "text",
            "diagram",
            "table",
            "json",
            "skip",
            "focus",
        ]

        for concept in required_concepts:
            assert concept.lower() in prompt.lower(), f"Missing concept: {concept}"

        # Check for specific document elements
        document_elements = ["title", "content", "references"]
        found_elements = sum(1 for elem in document_elements if elem.lower() in prompt.lower())
        assert found_elements >= 2, "System prompt should mention document elements"

    def test_content_prompts_coverage(self):
        """Test that all content types have prompts."""
        content_types = ["text", "diagram", "table", "mixed"]

        for content_type in content_types:
            prompt = get_content_specific_prompt(content_type)
            assert isinstance(prompt, str)
            assert len(prompt) > 0
