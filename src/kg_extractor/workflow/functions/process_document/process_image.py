"""Image processing module for handling image files."""

import base64
import os
from pathlib import Path


def process_image(image_path: str) -> str:
    """Process an image file directly.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded image string

    Raises:
        FileNotFoundError: If image file doesn't exist
        PermissionError: If image file cannot be read
        IOError: If image file cannot be processed
    """
    path = Path(image_path)

    # Validate file exists
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate file is readable
    if not os.access(image_path, os.R_OK):
        raise PermissionError(f"Cannot read image file: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        raise IOError(f"Failed to process image: {e}") from e
