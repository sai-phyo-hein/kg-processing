# KG Extractor

Extract structured content from documents using NVIDIA API for knowledge graph construction.

## Features

- Modern Python project structure with `src/` layout
- `uv` for fast dependency management
- NVIDIA API integration for intelligent content extraction
- **Multi-format support:**
  - Images: PNG, JPEG, WEBP, GIF, BMP
  - Documents: PDF, DOCX, PPTX, XLSX
- **Intelligent content processing:**
  - Text extraction with sentence restructuring
  - Detailed diagram and chart analysis
  - Table-to-JSON conversion
  - Automatic content filtering (skips TOC, references, etc.)
- Streaming and non-streaming API responses
- Multiple output formats (text, JSON)
- Content-type specific processing (text, diagrams, tables, mixed)
- Comprehensive error handling and retry logic
- Pre-configured development tools:
  - `pytest` for testing
  - `black` for code formatting
  - `ruff` for linting
  - `mypy` for type checking
- CLI entry point configured

## Project Structure

```
kg-extractor/
├── src/
│   └── kg_extractor/
│       ├── __init__.py
│       ├── main.py
│       ├── parser.py
│       └── input_processor.py
├── tests/
│   ├── test_main.py
│   ├── test_parser.py
│   └── test_input_processor.py
├── pyproject.toml
├── README.md
├── .gitignore
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` installed (https://github.com/astral-sh/uv)
- NVIDIA API key (get one from https://build.nvidia.com/)

### Installation

```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your NVIDIA API key:
```
NVIDIA_API_KEY=your_actual_api_key_here
```

### Usage

```bash
# Basic usage - extract content from an image
uv run kg-extractor image.jpg

# Process PDF documents
uv run kg-extractor document.pdf

# Process Word documents
uv run kg-extractor report.docx

# Process PowerPoint presentations
uv run kg-extractor presentation.pptx

# Process Excel spreadsheets
uv run kg-extractor data.xlsx

# Use streaming mode (images only)
uv run kg-extractor --stream image.png

# Focus on specific content types
uv run kg-extractor --content-type table spreadsheet.xlsx
uv run kg-extractor --content-type diagram chart.pdf
uv run kg-extractor --content-type text document.docx

# Get structured JSON output
uv run kg-extractor --format json --output result.json report.pdf

# Save output to file
uv run kg-extractor --output result.txt image.jpg

# Customize model and parameters
uv run kg-extractor --model google/gemma-3-27b-it --max-tokens 4096 --temperature 0.3 document.pdf

# Show help
uv run kg-extractor --help

# Show version
uv run kg-extractor --version
```

### Supported Formats

#### Image Formats
- PNG (.png)
- JPEG (.jpg, .jpeg)
- WEBP (.webp)
- GIF (.gif)
- BMP (.bmp)

#### Document Formats
- PDF (.pdf)
- Word documents (.docx, .doc)
- PowerPoint presentations (.pptx, .ppt)
- Excel spreadsheets (.xlsx, .xls)

### Content Processing Features

The extractor uses intelligent AI-powered processing to handle different types of content:

#### Text Processing
- Extracts and restructures text content
- Fixes broken sentences using context
- Maintains technical accuracy and meaning
- Improves readability while preserving intent

#### Diagram & Chart Analysis
- Provides detailed explanations of visual elements
- Identifies chart types (bar, line, pie, etc.)
- Explains data relationships and patterns
- Describes axes, labels, and legends
- Highlights trends and insights

#### Table Processing
- Converts tables to structured JSON format
- Preserves data types and structure
- Handles merged cells appropriately
- Includes table captions and summaries

#### Smart Content Filtering
- **Skips:** Title pages, table of contents, references, appendices
- **Focuses on:** Main content, methodology, results, conclusions

### API Parameters

- `--model`: NVIDIA model to use (default: google/gemma-3-27b-it)
- `--max-tokens`: Maximum tokens in response (default: 2048)
- `--temperature`: Sampling temperature, 0-2 (default: 0.20)
- `--top-p`: Nucleus sampling parameter, 0-1 (default: 0.70)
- `--content-type`: Content focus - text/diagram/table/mixed (default: mixed)
- `--format`: Output format - text/json (default: text)
- `--stream`: Use streaming API response (images only)
- `--output`: Save output to file instead of printing

## Development

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_main.py

# Run with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=kg_extractor
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Troubleshooting

### API Key Issues

If you see "NVIDIA_API_KEY environment variable not set":
1. Make sure you've created a `.env` file
2. Add your API key: `NVIDIA_API_KEY=your_key_here`
3. Ensure the `.env` file is in the project root

### File Processing Errors

- **File not found**: Check the file path is correct
- **Unsupported format**: Ensure file is one of the supported formats
- **Permission denied**: Check file read permissions
- **Corrupted file**: Try using a different file

### Document Processing Issues

- **Large documents**: Processing may take time for multi-page documents
- **Complex layouts**: Some complex layouts may not render perfectly
- **Password-protected files**: Not currently supported
- **Very large files**: Consider splitting into smaller files

### API Errors

- **Invalid API key**: Verify your NVIDIA API key is correct
- **Rate limit exceeded**: Wait before retrying
- **Network error**: Check your internet connection
- **Timeout error**: Large documents may require longer processing time

### Content Quality Issues

- **Incomplete extraction**: Try increasing `--max-tokens`
- **Poor text quality**: Ensure source documents are clear and readable
- **Missing diagrams**: Check that visual elements are high-resolution
- **Table parsing errors**: Complex tables may need manual review

## License

Add your license here.