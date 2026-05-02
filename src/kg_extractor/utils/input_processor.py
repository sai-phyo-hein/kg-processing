"""Input processor module for coordinating document processing."""

from pathlib import Path
from typing import List, Optional, Tuple

from kg_extractor.processors.docx_processor import process_docx
from kg_extractor.processors.image_processor import process_image
from kg_extractor.processors.pdf_processor import process_pdf
from kg_extractor.processors.pptx_processor import process_pptx
from kg_extractor.utils.prompts import get_system_prompt  # noqa: F401
from kg_extractor.processors.xlsx_processor import process_xlsx


class DocumentProcessor:
    """Process various document formats and convert to images."""

    SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    SUPPORTED_DOCUMENT_FORMATS = {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
    }

    def __init__(self, file_path: str):
        """Initialize DocumentProcessor with a file path.

        Args:
            file_path: Path to the document file
        """
        self.file_path = file_path
        self.file_type = self.get_file_type(file_path)

    def convert_to_images(self, pages: Optional[List[int]] = None) -> dict:
        """Convert document to images and return as dictionary.

        Args:
            pages: Optional list of page numbers to process (1-indexed)

        Returns:
            Dictionary mapping page numbers to base64 encoded images

        Raises:
            ValueError: If file format is not supported or pages are out of range
        """
        _, images = self.process_document(self.file_path, pages=pages)

        # Return as dictionary with page numbers as keys
        if pages is not None:
            # If specific pages requested, use those page numbers
            return {page_num: images[i] for i, page_num in enumerate(pages)}
        else:
            # Otherwise, use sequential page numbers starting from 1
            return {i + 1: img for i, img in enumerate(images)}

    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine the type of file.

        Args:
            file_path: Path to the file

        Returns:
            File type: 'image', 'pdf', 'docx', 'pptx', 'xlsx', or 'unknown'
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in DocumentProcessor.SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif suffix == ".pdf":
            return "pdf"
        elif suffix in {".docx", ".doc"}:
            return "docx"
        elif suffix in {".pptx", ".ppt"}:
            return "pptx"
        elif suffix in {".xlsx", ".xls"}:
            return "xlsx"
        else:
            return "unknown"

    @staticmethod
    def process_document(file_path: str, pages: Optional[List[int]] = None) -> Tuple[str, List[str]]:
        """Process any supported document format.

        Args:
            file_path: Path to the document file
            pages: Optional list of page numbers to process (1-indexed).
                   If None, all pages are processed.

        Returns:
            Tuple of (file_type, list of base64 encoded images)

        Raises:
            ValueError: If file format is not supported or requested pages are out of range
        """
        file_type = DocumentProcessor.get_file_type(file_path)

        if file_type == "unknown":
            raise ValueError(
                f"Unsupported file format. Supported formats: "
                f"Images ({', '.join(DocumentProcessor.SUPPORTED_IMAGE_FORMATS)}), "
                f"Documents ({', '.join(DocumentProcessor.SUPPORTED_DOCUMENT_FORMATS)})"
            )

        if file_type == "image":
            base64_image = process_image(file_path)
            return file_type, [base64_image]
        elif file_type == "pdf":
            images = process_pdf(file_path)
        elif file_type == "docx":
            images = process_docx(file_path)
        elif file_type == "pptx":
            images = process_pptx(file_path)
        elif file_type == "xlsx":
            images = process_xlsx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Filter pages if specified
        if pages is not None:
            total_pages = len(images)
            # Validate page numbers
            for page_num in pages:
                if page_num < 1 or page_num > total_pages:
                    raise ValueError(
                        f"Page {page_num} is out of range. Document has {total_pages} pages."
                    )
            # Filter images (pages is 1-indexed, list is 0-indexed)
            filtered_images = [images[page_num - 1] for page_num in pages]
            return file_type, filtered_images

        return file_type, images
