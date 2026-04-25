"""PDF processing module for handling PDF documents."""

import os
import tempfile
from typing import List

try:
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError as e:
    raise ImportError(f"Required dependencies not installed. Please install: {e}") from e


def process_pdf(pdf_path: str) -> List[str]:
    """Process PDF file and convert pages to images.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of base64 encoded page images

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        PermissionError: If PDF file cannot be read
        IOError: If PDF file cannot be processed
    """
    import base64

    # Validate file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Validate file is readable
    if not os.access(pdf_path, os.R_OK):
        raise PermissionError(f"Cannot read PDF file: {pdf_path}")

    base64_images = []

    try:
        # Open PDF using PyMuPDF for better rendering
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get page dimensions
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to base64
            buffered = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(buffered, format="PNG")
            buffered.close()

            with open(buffered.name, "rb") as f:
                image_data = f.read()
                base64_images.append(base64.b64encode(image_data).decode("utf-8"))

            os.unlink(buffered.name)

        doc.close()
    except Exception as e:
        raise IOError(f"Failed to process PDF: {e}") from e

    return base64_images
