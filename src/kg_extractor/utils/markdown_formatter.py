"""Markdown formatter for document analysis results."""

import json
from pathlib import Path
from typing import Any, Dict


def format_result_to_markdown(result: Dict[str, Any], file_path: str) -> str:
    """Format document analysis result to markdown.

    Args:
        result: Analysis result from API
        file_path: Original file path for reference

    Returns:
        Formatted markdown string
    """
    md_lines = []

    # Header
    file_name = Path(file_path).name
    md_lines.append(f"# Document Analysis: {file_name}\n")
    md_lines.append(f"**File Type:** {result.get('file_type', 'unknown')}\n")
    md_lines.append(f"**Total Pages:** {result.get('total_pages', 0)}\n")
    md_lines.append(
        f"**Processing Model:** {result.get('metadata', {}).get('processing_model', 'unknown')}\n"
    )
    md_lines.append(
        f"**Content Type:** {result.get('metadata', {}).get('content_type', 'mixed')}\n"
    )
    md_lines.append("---\n")

    # Start flag for all extracted content
    md_lines.append("<start>\n")

    # Process each page
    for page in result.get("pages", []):
        page_num = page.get("page_number", 1)
        content = page.get("content", {})

        # Text content - handle both string and JSON string formats
        if "text" in content and content["text"]:
            # Removed "### Text Content" header as requested
            text_data = content["text"]

            # Try to parse if it's a JSON string wrapped in markdown code blocks
            if isinstance(text_data, str):
                try:
                    # Remove markdown code blocks if present
                    cleaned_text = text_data.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]  # Remove ```json
                    elif cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:]  # Remove ```
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]  # Remove trailing ```

                    cleaned_text = cleaned_text.strip()

                    # Check if it looks like JSON
                    if cleaned_text.startswith("{") or cleaned_text.startswith("["):
                        parsed = json.loads(cleaned_text)
                        if isinstance(parsed, dict) and "content" in parsed:
                            # Extract the actual content from nested structure
                            nested_content = parsed["content"]
                            if isinstance(nested_content, dict):
                                if "text" in nested_content:
                                    # Handle both string and list text
                                    text_value = nested_content["text"]
                                    if isinstance(text_value, str):
                                        md_lines.append(text_value)
                                    elif isinstance(text_value, list):
                                        for text_item in text_value:
                                            md_lines.append(f"- {text_item}")
                                    else:
                                        md_lines.append(str(text_value))
                                if "diagrams" in nested_content and nested_content["diagrams"]:
                                    md_lines.append("\n**Diagrams:**")
                                    for diagram in nested_content["diagrams"]:
                                        md_lines.append(
                                            f"- {diagram.get('type', 'Unknown')}: "
                                            f"{diagram.get('description', '')}"
                                        )
                                if "tables" in nested_content and nested_content["tables"]:
                                    md_lines.append("\n**Tables:**")
                                    for table in nested_content["tables"]:
                                        md_lines.append(
                                            f"- {table.get('title', 'Untitled')}: "
                                            f"{table.get('summary', '')}"
                                        )
                            else:
                                md_lines.append(str(nested_content))
                        else:
                            md_lines.append(text_data)
                    else:
                        md_lines.append(text_data)
                except json.JSONDecodeError:
                    md_lines.append(text_data)
            elif isinstance(text_data, list):
                for text_item in text_data:
                    md_lines.append(f"- {text_item}")
            else:
                md_lines.append(str(text_data))
            md_lines.append("")

        # Diagrams (from original content structure, not nested)
        if "diagrams" in content and content["diagrams"]:
            # Removed "### Diagrams and Charts" header as requested
            for i, diagram in enumerate(content["diagrams"], 1):
                md_lines.append(f"#### Diagram {i}\n")
                if "type" in diagram:
                    md_lines.append(f"**Type:** {diagram['type']}\n")
                if "description" in diagram:
                    md_lines.append(f"**Description:** {diagram['description']}\n")
                if "data_insights" in diagram:
                    md_lines.append(f"**Data Insights:** {diagram['data_insights']}\n")
                md_lines.append("")

        # Tables (from original content structure, not nested)
        if "tables" in content and content["tables"]:
            # Removed "### Tables" header as requested
            for i, table in enumerate(content["tables"], 1):
                md_lines.append(f"#### Table {i}\n")
                if "title" in table:
                    md_lines.append(f"**Title:** {table['title']}\n")
                if "summary" in table:
                    md_lines.append(f"**Summary:** {table['summary']}\n")
                if "structure" in table:
                    md_lines.append("**Data:**\n")
                    md_lines.append("```json")
                    md_lines.append(json.dumps(table["structure"], indent=2))
                    md_lines.append("```")
                md_lines.append("")

        # Metadata
        if "metadata" in content:
            # Removed "### Page Metadata" header as requested
            metadata = content["metadata"]
            md_lines.append(f"- **Has Text:** {metadata.get('has_text', False)}")
            md_lines.append(f"- **Has Diagrams:** {metadata.get('has_diagrams', False)}")
            md_lines.append(f"- **Has Tables:** {metadata.get('has_tables', False)}")
            if "content_quality" in metadata:
                md_lines.append(f"- **Content Quality:** {metadata['content_quality']}")
            md_lines.append("")

        md_lines.append("---\n")

    # End flag for all extracted content
    md_lines.append("</end>\n")

    return "\n".join(md_lines)


def save_markdown_result(result: Dict[str, Any], file_path: str, output_dir: str = None) -> str:
    """Save analysis result as markdown file.

    Args:
        result: Analysis result from API
        file_path: Original file path for reference
        output_dir: Directory to save markdown file (default: project root/output)

    Returns:
        Path to saved markdown file
    """
    # Set default output directory to project root/output
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    input_file = Path(file_path)
    output_filename = f"{input_file.stem}_analysis.md"
    output_file = output_path / output_filename

    # Format and save markdown
    markdown_content = format_result_to_markdown(result, file_path)
    output_file.write_text(markdown_content, encoding="utf-8")

    return str(output_file)


def format_text_to_markdown(text: str, file_path: str) -> str:
    """Format plain text result to markdown.

    Args:
        text: Extracted text content
        file_path: Original file path for reference

    Returns:
        Formatted markdown string
    """
    md_lines = []

    # Header
    file_name = Path(file_path).name
    md_lines.append(f"# Document Analysis: {file_name}\n")
    md_lines.append("---\n")
    md_lines.append(text)

    return "\n".join(md_lines)


def save_text_markdown(text: str, file_path: str, output_dir: str = None) -> str:
    """Save text result as markdown file.

    Args:
        text: Extracted text content
        file_path: Original file path for reference
        output_dir: Directory to save markdown file (default: project root/output)

    Returns:
        Path to saved markdown file
    """
    # Set default output directory to project root/output
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    input_file = Path(file_path)
    output_filename = f"{input_file.stem}_analysis.md"
    output_file = output_path / output_filename

    # Format and save markdown
    markdown_content = format_text_to_markdown(text, file_path)
    output_file.write_text(markdown_content, encoding="utf-8")

    return str(output_file)
