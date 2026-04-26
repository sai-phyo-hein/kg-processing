# KG Extractor

Extract structured content from documents using AI APIs for knowledge graph construction, with intelligent agent-based processing and markdown editing capabilities.

## Features

- Modern Python project structure with `src/` layout
- `uv` for fast dependency management
- **Multi-API provider support:**
  - OpenAI (GPT models)
  - NVIDIA API (vision models)
  - OpenRouter (multiple free models)
  - Groq (fast inference)
- **Multi-format support:**
  - Images: PNG, JPEG, WEBP, GIF, BMP
  - Documents: PDF, DOCX, PPTX, XLSX
- **Intelligent content processing:**
  - Text extraction with sentence restructuring
  - Detailed diagram and chart analysis
  - Table-to-JSON conversion
  - Automatic content filtering (skips TOC, references, etc.)
- **LangChain Agent Integration:**
  - Document processing tools
  - Markdown file editing (read, edit, search, list)
  - Interactive and single-task modes
  - Multi-step reasoning and tool use
- **Multiple output formats:** text, JSON, markdown
- **Modular architecture:** Separated concerns for maintainability
- Streaming and non-streaming API responses
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
│       ├── main.py              # CLI entry point
│       ├── parser.py            # API integration (NVIDIA, OpenRouter)
│       ├── agent.py             # LangChain agent with tools
│       ├── input_processor.py   # Main document processor
│       ├── prompts.py           # System and content prompts
│       ├── markdown_formatter.py # Markdown output formatting
│       ├── image_processor.py   # Image processing
│       ├── pdf_processor.py     # PDF processing
│       ├── docx_processor.py    # Word document processing
│       ├── pptx_processor.py    # PowerPoint processing
│       └── xlsx_processor.py    # Excel spreadsheet processing
├── tests/
│   ├── test_main.py
│   ├── test_parser.py
│   └── test_input_processor.py
├── output/                      # Generated markdown files
├── pyproject.toml
├── README.md
├── .gitignore
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `uv` installed (https://github.com/astral-sh/uv)
- API keys for one or more providers:
  - OpenAI API key (get one from https://platform.openai.com/api-keys)
  - NVIDIA API key (get one from https://build.nvidia.com/)
  - OpenRouter API key (get one from https://openrouter.ai/keys)
  - Groq API key (get one from https://console.groq.com/keys)

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

2. Edit `.env` and add your API keys:
```
# OpenAI (recommended for agent mode)
OPENAI_API_KEY=your_openai_api_key_here

# NVIDIA API
NVIDIA_API_KEY=your_nvidia_api_key_here

# OpenRouter API
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Groq API
GROQ_API_KEY=your_groq_api_key_here
```

### Usage

#### Document Processing Mode

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

# Get markdown output (saved to output/ directory)
uv run kg-extractor --format markdown report.pdf

# Save output to file
uv run kg-extractor --output result.txt image.jpg

# Use different API providers
uv run kg-extractor --provider openai --model gpt-4o-mini document.pdf
uv run kg-extractor --provider nvidia --model microsoft/phi-4-multimodal-instruct image.jpg
uv run kg-extractor --provider openrouter --model openai/gpt-oss-120b:free document.pdf

# Customize model and parameters
uv run kg-extractor --model google/gemma-3-27b-it --max-tokens 4096 --temperature 0.3 document.pdf

# Show help
uv run kg-extractor --help

# Show version
uv run kg-extractor --version
```

#### Agent Mode

The agent mode provides intelligent document processing and markdown editing capabilities using LangChain:

```bash
# Interactive agent mode
uv run kg-extractor --agent

# Single task mode
uv run kg-extractor --agent --agent-task "List all markdown files in the output directory"

# Process documents with agent
uv run kg-extractor --agent --agent-task "Process document.pdf and save as markdown"

# Complex multi-step tasks
uv run kg-extractor --agent --agent-task "Read the first 10 lines of output/test_analysis.md and then search for 'Thailand' in the file"

# Edit markdown files
uv run kg-extractor --agent --agent-task "Edit line 5 of output/test_analysis.md to say 'Updated content'"

# Search and analyze content
uv run kg-extractor --agent --agent-task "Search for 'EBITDA' in output/test_analysis.md and summarize the findings"

# Use specific model for agent
uv run kg-extractor --agent --agent-model gpt-4o-mini --agent-task "List all markdown files"
```

**Available Agent Tools:**

1. **process_document_tool** - Process documents (PDF, DOCX, PPTX, XLSX, images) and extract structured content
2. **read_markdown_file** - Read markdown files line by line
3. **edit_markdown_file** - Edit markdown files at specific lines (replace, insert, append)
4. **search_markdown_content** - Search for content in markdown files
5. **list_markdown_files** - List all markdown files in a directory

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

## Architecture

### Modular Design

The project follows a modular architecture with clear separation of concerns:

#### Core Modules

- **`parser.py`** - API integration layer
  - Handles communication with NVIDIA, OpenRouter, and OpenAI APIs
  - Manages authentication and error handling
  - Supports streaming and non-streaming responses

- **`agent.py`** - LangChain agent integration
  - Provides intelligent document processing capabilities
  - Implements tool-based architecture for complex tasks
  - Supports multiple AI providers

- **`input_processor.py`** - Main document processor
  - Coordinates document processing workflow
  - Delegates to format-specific processors
  - Manages file type detection

#### Format-Specific Processors

- **`prompts.py`** - Prompt management
  - System prompts for document analysis
  - Content-specific prompts (text/diagram/table/mixed)

- **`image_processor.py`** - Image processing
  - Direct image file handling
  - Base64 encoding and validation

- **`pdf_processor.py`** - PDF processing
  - PDF rendering using PyMuPDF
  - Page-by-page image generation

- **`docx_processor.py`** - Word document processing
  - Text extraction from paragraphs
  - Document structure analysis

- **`pptx_processor.py`** - PowerPoint processing
  - Slide content extraction
  - Visual element processing

- **`xlsx_processor.py`** - Excel processing
  - Sheet data extraction
  - Table structure analysis

- **`markdown_formatter.py`** - Output formatting
  - Converts API results to markdown
  - Handles nested JSON structures
  - Manages file output

### Agent Tools

The LangChain agent provides five main tools:

1. **Document Processing** - Extract structured content from various file formats
2. **File Reading** - Read markdown files line by line with range support
3. **File Editing** - Edit markdown files at specific lines (replace/insert/append)
4. **Content Search** - Search for terms in markdown files with case sensitivity options
5. **File Listing** - List and analyze markdown files in directories

### Data Flow

```
User Input → CLI → Main Module
                ↓
        [Document Processing] or [Agent Mode]
                ↓
    Format-Specific Processors → API Integration
                ↓
        Content Extraction & Analysis
                ↓
    Output Formatting (Text/JSON/Markdown)
```

## Examples

### Document Analysis Workflow

```bash
# Process a PDF document and save as markdown
uv run kg-extractor --format markdown report.pdf

# The output will be saved to output/report_analysis.md
```

### Agent-Based Content Analysis

```bash
# Interactive session to analyze multiple documents
uv run kg-extractor --agent

# In the agent session, you can:
# - Process documents: "Process document.pdf and save as markdown"
# - Analyze content: "Read output/document_analysis.md and summarize the key findings"
# - Search for terms: "Search for 'revenue' in all markdown files"
# - Edit files: "Edit line 10 of output/summary.md to add more details"
```

### Batch Processing with Agent

```bash
# Process multiple documents and analyze results
uv run kg-extractor --agent --agent-task "Process all PDF files in the current directory and create a summary of their contents"
```

### Content Extraction and Analysis

```bash
# Extract tables from a spreadsheet
uv run kg-extractor --content-type table --format json data.xlsx

# Analyze diagrams in a presentation
uv run kg-extractor --content-type diagram presentation.pptx

# Extract and restructure text from a document
uv run kg-extractor --content-type text report.docx
```

### Advanced Agent Workflows

```bash
# Complex multi-step analysis
uv run kg-extractor --agent --agent-task "Process financial_report.pdf, then search the output for 'EBITDA', and finally create a summary of the financial metrics found"

# Document comparison
uv run kg-extractor --agent --agent-task "Read the first 20 lines of output/report1.md and output/report2.md, then compare their key differences"

# Content extraction and formatting
uv run kg-extractor --agent --agent-task "Search for all tables in output/analysis.md, extract their data, and format it as a summary"
```

### API Parameters

#### Document Processing Mode

- `--provider`: API provider - nvidia/openrouter (default: nvidia)
- `--model`: Model to use (default varies by provider)
- `--max-tokens`: Maximum tokens in response (default: 2048)
- `--temperature`: Sampling temperature, 0-2 (default: 0.20)
- `--top-p`: Nucleus sampling parameter, 0-1 (default: 0.70)
- `--content-type`: Content focus - text/diagram/table/mixed (default: mixed)
- `--format`: Output format - text/json/markdown (default: text)
- `--stream`: Use streaming API response (images only)
- `--output`: Save output to file instead of printing

#### Agent Mode

- `--agent`: Enable agent mode
- `--agent-model`: Model for agent (default: gpt-4o-mini)
- `--agent-task`: Single task for agent (non-interactive mode)

### Recommended Models

#### OpenAI (Best for Agent Mode)
- `gpt-4o-mini` - Fast, cost-effective, excellent tool calling
- `gpt-4o` - Higher quality, more expensive

#### NVIDIA (Good for Vision Tasks)
- `microsoft/phi-4-multimodal-instruct` - Excellent for document analysis
- `google/gemma-3-27b-it` - Good general purpose model

#### OpenRouter (Free Options)
- `openai/gpt-oss-120b:free` - Good tool calling support
- `google/gemma-4-31b-it:free` - May have rate limits

#### Groq (Fast Inference)
- `openai/gpt-oss-120b` - Very fast inference with Groq

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

# Run specific test
uv run pytest tests/test_parser.py::test_api_key_validation
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Fix linting issues automatically
uv run ruff check --fix src/ tests/
```

### Testing Agent Functionality

```bash
# Test agent with simple task
uv run kg-extractor --agent --agent-task "Say hello"

# Test agent with document processing
uv run kg-extractor --agent --agent-task "List all markdown files in the output directory"

# Test agent with file operations
uv run kg-extractor --agent --agent-task "Read the first 5 lines of output/test_analysis.md"
```

### Module Development

When adding new functionality:

1. **Create format-specific processor** (for new file types)
   - Follow the pattern in `pdf_processor.py`
   - Implement `process_<format>()` function
   - Add to `input_processor.py`

2. **Add new agent tools** (for new capabilities)
   - Create tool function with `@tool` decorator
   - Add to tools list in `agent.py`
   - Update system prompt if needed

3. **Add new prompts** (for different content types)
   - Add to `prompts.py`
   - Update `get_content_specific_prompt()` function

4. **Update tests**
   - Add tests for new functionality
   - Ensure existing tests still pass
   - Update test fixtures if needed

## Performance Considerations

### Document Processing

- **Large PDFs**: Multi-page documents may take significant time to process
- **High-resolution images**: Larger images require more API tokens and processing time
- **Complex layouts**: Documents with complex formatting may need longer processing

### API Usage

- **Token limits**: Be mindful of token limits when processing large documents
- **Rate limiting**: Some providers have rate limits, especially free tiers
- **Cost optimization**: Use appropriate models for different tasks
  - Use faster/cheaper models for simple tasks
  - Use higher-quality models for complex analysis

### Agent Mode

- **Tool calling overhead**: Each tool call adds latency
- **Multi-step tasks**: Complex tasks may require multiple API calls
- **Context management**: Long conversations may hit context limits

## Best Practices

### Document Processing

1. **Start with appropriate content type**
   - Use `--content-type table` for spreadsheets
   - Use `--content-type diagram` for presentations with charts
   - Use `--content-type text` for text-heavy documents

2. **Choose the right output format**
   - Use `--format markdown` for readable results
   - Use `--format json` for programmatic processing
   - Use `--format text` for simple extraction

3. **Optimize token usage**
   - Adjust `--max-tokens` based on document complexity
   - Use streaming mode for real-time feedback on large files

### Agent Usage

1. **Be specific with tasks**
   - Clear instructions lead to better results
   - Break complex tasks into smaller steps

2. **Use appropriate models**
   - OpenAI models work best for agent mode
   - Consider cost vs. quality trade-offs

3. **Handle errors gracefully**
   - Agent will retry on transient failures
   - Provide fallback instructions for robustness

## Future Enhancements

Potential areas for expansion:

- **Additional file formats**: Support for more document types
- **Batch processing**: Process multiple files in parallel
- **Advanced agent tools**: More sophisticated editing and analysis capabilities
- **Integration options**: API endpoints for programmatic access
- **Custom prompts**: User-defined prompts for specific use cases
- **Output templates**: Customizable output formats and templates
- **Performance optimization**: Caching and incremental processing
- **Collaboration features**: Multi-user support and sharing

## Troubleshooting

### API Key Issues

If you see "API_KEY environment variable not set":
1. Make sure you've created a `.env` file
2. Add the appropriate API key for your provider:
   - `OPENAI_API_KEY` for OpenAI
   - `NVIDIA_API_KEY` for NVIDIA
   - `OPENROUTER_API_KEY` for OpenRouter
   - `GROQ_API_KEY` for Groq
3. Ensure the `.env` file is in the project root

### Agent Mode Issues

**Agent not calling tools:**
- Some models have limited tool-calling support
- Use OpenAI models (gpt-4o-mini) for best tool support
- Check that the model supports function calling

**Rate limiting with free models:**
- OpenRouter free models may have rate limits
- Try using a different model or provider
- Consider using OpenAI for more reliable access

**Agent responses are empty:**
- Check your API key is valid
- Ensure the model is available and working
- Try with a simpler task first

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

- **Invalid API key**: Verify your API key is correct for the chosen provider
- **Rate limit exceeded**: Wait before retrying or switch providers
- **Network error**: Check your internet connection
- **Timeout error**: Large documents may require longer processing time
- **Model not found**: Check the model name is correct for the provider

### Content Quality Issues

- **Incomplete extraction**: Try increasing `--max-tokens`
- **Poor text quality**: Ensure source documents are clear and readable
- **Missing diagrams**: Check that visual elements are high-resolution
- **Table parsing errors**: Complex tables may need manual review

### Markdown Output Issues

- **Empty markdown files**: Check that the API returned valid content
- **Malformed JSON**: Some models may return incomplete JSON structures
- **Missing content**: Try with a different model or content type

## License

Add your license here.