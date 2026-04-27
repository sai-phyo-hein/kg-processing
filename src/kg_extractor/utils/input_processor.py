"""Input processor module for coordinating document processing."""

from pathlib import Path
from typing import List, Tuple

from kg_extractor.processors.docx_processor import process_docx
from kg_extractor.processors.image_processor import process_image
from kg_extractor.processors.pdf_processor import process_pdf
from kg_extractor.processors.pptx_processor import process_pptx
from kg_extractor.utils.prompts import get_content_specific_prompt, get_system_prompt  # noqa: F401
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
    def process_document(file_path: str) -> Tuple[str, List[str]]:
        """Process any supported document format.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (file_type, list of base64 encoded images)

        Raises:
            ValueError: If file format is not supported
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
            return file_type, images
        elif file_type == "docx":
            images = process_docx(file_path)
            return file_type, images
        elif file_type == "pptx":
            images = process_pptx(file_path)
            return file_type, images
        elif file_type == "xlsx":
            images = process_xlsx(file_path)
            return file_type, images
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
