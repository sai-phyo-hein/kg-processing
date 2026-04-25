"""Word document processing module for handling DOCX files."""

import os
import tempfile
from typing import List

try:
    from docx import Document
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise ImportError(f"Required dependencies not installed. Please install: {e}") from e


def _create_text_image(text: str, title: str = "") -> Image.Image:
    """Create an image from text content.

    Args:
        text: Text content to render
        title: Optional title for the image

    Returns:
        PIL Image with rendered text
    """
    # Create a white image
    width, height = 800, 600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except (IOError, OSError):
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Draw title if provided
    if title:
        draw.text((10, 10), title, fill="black", font=title_font)

    # Draw text content (simplified - in production, you'd want proper text wrapping)
    y_offset = 50
    for line in text.split("\n"):
        if y_offset < height - 30:
            draw.text((10, y_offset), line, fill="black", font=font)
            y_offset += 25

    return img


def process_docx(docx_path: str) -> List[str]:
    """Process DOCX file and convert pages to images.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        List of base64 encoded page images

    Raises:
        FileNotFoundError: If DOCX file doesn't exist
        PermissionError: If DOCX file cannot be read
        IOError: If DOCX file cannot be processed
    """
    import base64

    # Validate file exists
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    # Validate file is readable
    if not os.access(docx_path, os.R_OK):
        raise PermissionError(f"Cannot read DOCX file: {docx_path}")

    base64_images = []

    try:
        # For DOCX, we'll extract text and create images
        # This is a simplified approach - in production, you might want to use
        # a more sophisticated method like LibreOffice headless conversion
        doc = Document(docx_path)

        # Create images from document content
        # This is a placeholder - actual implementation would need proper rendering
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                # Create a simple text-based image
                img = _create_text_image(para.text, f"Page {i + 1}")

                buffered = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(buffered, format="PNG")
                buffered.close()

                with open(buffered.name, "rb") as f:
                    image_data = f.read()
                    base64_images.append(base64.b64encode(image_data).decode("utf-8"))

                os.unlink(buffered.name)
    except Exception as e:
        raise IOError(f"Failed to process DOCX: {e}") from e

    return base64_images
