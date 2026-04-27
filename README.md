# KG Extractor

Extract structured content from documents using AI APIs for knowledge graph construction, with intelligent agent-based processing, LLM-based semantic chunking, and OpenIE triple extraction with causal discovery.

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
- **Three-step workflow:**
  1. Document processing with AI-powered content extraction
  2. LLM-based semantic chunking based on topic shifts
  3. Knowledge graph triple extraction with OpenIE
- **Intelligent content processing:**
  - Text extraction with sentence restructuring
  - Detailed diagram and chart analysis
  - Table-to-JSON conversion
  - Automatic content filtering (skips TOC, references, etc.)
- **Semantic chunking:**
  - LLM-based topic boundary detection (not embedding-based)
  - Configurable granularity via similarity threshold
  - Semantic coherence preservation
- **Knowledge graph triple extraction:**
  - OpenIE (Open Information Extraction) with zero-shot discovery
  - Two-level predicate hierarchy (Specific Predicate → Relationship Class)
  - Causal discovery with triggered_by, mechanism, and causal_weight
  - Temporal metadata extraction (validFrom, validTo, status)
  - Evidence-based extraction with exact text quotes
  - Parallel processing for efficient extraction
- **LangChain Agent Integration:**
  - Document processing tools
  - Markdown file editing (read, edit, search, list)
  - Triple extraction tools
  - Interactive and single-task modes
  - Multi-step reasoning and tool use
- **Multiple output formats:** text, JSON, markdown
- **Modular architecture:** tools/, utils/, processors/ structure
- Streaming and non-streaming API responses
- Content-type specific processing (text, diagrams, tables, mixed)
- Comprehensive error handling and retry logic
- Pre-configured development tools:
  - `pytest` for testing
  - `black` for code formatting
  - `ruff` for linting
  - `mypy` for type checking
- CLI entry point configured

## Key Technical Concepts

### OpenIE (Open Information Extraction)

OpenIE is a paradigm for extracting relations from text without requiring a predefined schema:

- **Zero-shot discovery**: Discovers predicates and categories dynamically from source text
- **No predefined ontology**: Builds an emergent ontology based on the text
- **Flexible extraction**: Can handle any domain without prior knowledge
- **Two-level hierarchy**: Specific predicates grouped into broad relationship classes

### LLM-based Semantic Chunking

Unlike traditional embedding-based chunking, this system uses LLM comprehension:

- **Topic shift detection**: LLM reads content and identifies where topics change
- **Semantic coherence**: Chunks are based on meaning, not just similarity scores
- **Context awareness**: LLM understands the content and creates meaningful boundaries
- **Configurable granularity**: Similarity threshold controls chunk size (repurposed from embedding-based approach)

### Causal Discovery

The system identifies causal relationships in the text:

- **Trigger identification**: Events or conditions that cause relationships
- **Mechanism extraction**: How the trigger leads to the relationship
- **Confidence scoring**: Causal weight (0.0-1.0) indicates confidence in the causal link
- **Evidence-based**: All causal links are supported by text evidence

### Parallel Processing

The system uses parallel processing for efficiency:

- **Document pages**: Multi-page documents are processed in parallel (max_workers=5)
- **Chunk processing**: Chunks are processed in parallel for triple extraction
- **Ordered results**: Results are sorted to maintain original order
- **Error handling**: Failed chunks/pages are handled gracefully without stopping the workflow

### Similarity Threshold and Chunking Granularity

The `similarity_threshold` parameter controls chunk granularity in LLM-based semantic chunking:

- **0.0 - 0.3 (Very fine-grained)**: Many small chunks, detects even subtle topic shifts
- **0.3 - 0.5 (Fine-grained)**: Smaller chunks, detects clear topic changes
- **0.5 - 0.7 (Medium-grained)**: Moderate chunks, detects significant topic shifts (default)
- **0.7 - 1.0 (Coarse-grained)**: Larger chunks, only detects major topic changes

**Note**: Despite the name, this parameter does not use similarity scores or embeddings. It's a configuration parameter that instructs the LLM on how granular to be when detecting topic boundaries.

## Use Cases

### Knowledge Graph Construction

Build knowledge graphs from unstructured documents:

- Extract entities and relationships from research papers
- Create causal models from business documents
- Build domain-specific ontologies from technical documentation
- Map organizational structures from company documents

### Document Analysis

Analyze complex documents for insights:

- Extract key findings from financial reports
- Identify causal relationships in case studies
- Summarize technical documentation
- Analyze business processes and workflows

### Content Organization

Organize large documents into manageable chunks:

- Create semantically coherent sections from long documents
- Identify topic boundaries in research papers
- Split technical documentation into logical sections
- Organize meeting notes by topic

### Information Extraction

Extract structured information from unstructured text:

- Extract entities and their relationships
- Identify temporal information and events
- Discover causal mechanisms and triggers
- Extract evidence for claims and assertions

## Project Structure

```
kg-extractor/
├── src/
│   └── kg_extractor/
│       ├── __init__.py
│       ├── main.py              # CLI entry point
│       ├── tools/               # Agent tools and workflow orchestration
│       │   ├── __init__.py
│       │   ├── agent.py         # LangChain agent with tools
│       │   └── workflow.py      # Three-step workflow orchestrator
│       ├── utils/               # Utility modules
│       │   ├── __init__.py
│       │   ├── parser.py        # API integration (NVIDIA, OpenRouter, Groq, OpenAI)
│       │   ├── prompts.py       # System and content prompts
│       │   ├── markdown_formatter.py # Markdown output formatting
│       │   ├── input_processor.py # Main document processor
│       │   ├── semantic_chunker.py # LLM-based semantic chunking
│       │   └── triple_extractor.py # OpenIE triple extraction
│       └── processors/          # Format-specific processors
│           ├── __init__.py
│           ├── image_processor.py   # Image processing
│           ├── pdf_processor.py     # PDF processing
│           ├── docx_processor.py    # Word document processing
│           ├── pptx_processor.py    # PowerPoint processing
│           └── xlsx_processor.py    # Excel spreadsheet processing
├── tests/
│   ├── test_main.py
│   ├── test_parser.py
│   └── test_input_processor.py
├── output/                      # Generated markdown, chunks, and triples files
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

#### Quick Start

```bash
# Run the complete workflow (recommended for first-time users)
uv run kg-extractor workflow document.pdf

# This will:
# 1. Process the document and extract structured content
# 2. Chunk the content semantically based on topic shifts
# 3. Extract knowledge graph triples with causal discovery
# 4. Save results to output/ directory
```

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
6. **extract_triples_tool** - Extract knowledge graph triples from markdown content

#### Workflow Mode

The workflow mode provides a complete three-step pipeline for document processing, semantic chunking, and knowledge graph triple extraction:

```bash
# Run the complete workflow
uv run kg-extractor workflow document.pdf

# Use different providers for each step
uv run kg-extractor workflow document.pdf --provider nvidia --chunking-llm-provider openai --triplet-llm-provider openai

# Customize chunking parameters
uv run kg-extractor workflow document.pdf --similarity-threshold 0.3 --min-chunk-size 50 --max-chunk-size 800

# Use different models
uv run kg-extractor workflow document.pdf --model microsoft/phi-4-multimodal-instruct --chunking-llm-model gpt-4o-mini --triplet-llm-model gpt-4o-mini

# Focus on specific content types
uv run kg-extractor workflow document.pdf --content-type diagram
uv run kg-extractor workflow document.pdf --content-type table
uv run kg-extractor workflow document.pdf --content-type text
```

**Workflow Steps:**

1. **Document Processing** - Extract structured content from the input document using AI vision models
2. **Semantic Chunking** - LLM-based chunking that detects topic shifts and creates semantically coherent chunks
3. **Triple Extraction** - OpenIE extraction that discovers entities, relationships, and causal links with zero-shot ontology discovery

**Output Files:**

- `output/<filename>_analysis.md` - Structured markdown from document processing
- `output/<filename>_chunks.json` - Semantically chunked content with chunk_id and content
- `output/<filename>_triples.json` - Knowledge graph triples with subject-predicate-object structure, relationship classes, and causal links

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

#### Semantic Chunking

The semantic chunking module uses LLM-based analysis to create semantically coherent chunks:

- **LLM-based topic detection**: Uses LLM comprehension to identify topic shifts (not embedding-based)
- **Configurable granularity**: Adjust similarity threshold to control chunk size
  - Lower threshold (0.0-0.3): Very fine-grained, many small chunks
  - Medium threshold (0.3-0.7): Balanced chunk sizes
  - Higher threshold (0.7-1.0): Coarse-grained, fewer large chunks
- **Semantic coherence**: Each chunk discusses a single topic or theme
- **Topic boundary detection**: Identifies where content shifts to new subjects

#### Knowledge Graph Triple Extraction

The triple extraction module implements OpenIE (Open Information Extraction) with zero-shot discovery:

- **Zero-shot ontology discovery**: Discovers predicates and categories dynamically from source text
- **Two-level predicate hierarchy**:
  - **Level 1 (Specific Predicate)**: Granular verb representing the action (e.g., `ISSUED_DEBT_INSTRUMENT`)
  - **Level 2 (Relationship Class)**: Broad conceptual category (e.g., `CAPITAL_MARKETS`)
- **Causal discovery**: Identifies causal relationships with:
  - `triggered_by`: Event or condition that caused the relationship
  - `mechanism`: How the trigger led to the relationship
  - `causal_weight`: Confidence score (0.0-1.0) for the causal link
- **Temporal metadata**: Extracts `validFrom`, `validTo`, and `status` (Current/Archived)
- **Evidence-based extraction**: Includes exact text quotes for each triple
- **Parallel processing**: Processes chunks in parallel for efficient extraction
- **Entity resolution**: Maintains consistent naming across the document

**Triple Structure:**

```json
{
  "subject": {
    "name": "Entity Name",
    "type": "Person/Organization/Concept"
  },
  "predicate": "SPECIFIC_VERB_IN_ALL_CAPS",
  "object": {
    "name": "Entity Name",
    "type": "Person/Organization/Concept"
  },
  "properties": {
    "relationship_class": "BROAD_CATEGORY",
    "status": "Current | Archived",
    "evidence_quote": "Exact text from source",
    "validFrom": "YYYY-MM-DD | null",
    "validTo": "YYYY-MM-DD | null",
    "causal_link": {
      "triggered_by": "Event or condition | null",
      "mechanism": "How it happened | null",
      "causal_weight": 0.0-1.0 | null
    }
  }
}
```

## Architecture

### Modular Design

The project follows a modular architecture with clear separation of concerns:

#### Core Modules

- **`tools/agent.py`** - LangChain agent integration
  - Provides intelligent document processing capabilities
  - Implements tool-based architecture for complex tasks
  - Supports multiple AI providers
  - Includes triple extraction tools

- **`tools/workflow.py`** - Three-step workflow orchestrator
  - Coordinates document processing, chunking, and triple extraction
  - Manages parallel processing for efficiency
  - Handles error recovery and result aggregation

- **`utils/parser.py`** - API integration layer
  - Handles communication with NVIDIA, OpenRouter, Groq, and OpenAI APIs
  - Manages authentication and error handling
  - Supports streaming and non-streaming responses
  - Implements parallel processing for document pages

- **`utils/semantic_chunker.py`** - LLM-based semantic chunking
  - Uses LLM to analyze content and detect topic boundaries
  - Creates semantically coherent chunks based on topic shifts
  - Configurable granularity via similarity threshold
  - Not embedding-based - uses LLM comprehension

- **`utils/triple_extractor.py`** - OpenIE triple extraction
  - Extracts knowledge graph triples using LLM analysis
  - Implements zero-shot ontology discovery
  - Supports two-level predicate hierarchy
  - Parallel processing of chunks for efficiency
  - Causal link discovery with confidence weights

- **`utils/input_processor.py`** - Main document processor
  - Coordinates document processing workflow
  - Delegates to format-specific processors
  - Manages file type detection

- **`utils/prompts.py`** - Prompt management
  - System prompts for document analysis
  - Content-specific prompts (text/diagram/table/mixed)
  - Chunking prompts for LLM-based topic detection
  - Triple extraction prompts for OpenIE discovery

#### Format-Specific Processors

- **`processors/image_processor.py`** - Image processing
  - Direct image file handling
  - Base64 encoding and validation

- **`processors/pdf_processor.py`** - PDF processing
  - PDF rendering using PyMuPDF
  - Page-by-page image generation
  - Parallel processing support

- **`processors/docx_processor.py`** - Word document processing
  - Text extraction from paragraphs
  - Document structure analysis

- **`processors/pptx_processor.py`** - PowerPoint processing
  - Slide content extraction
  - Visual element processing

- **`processors/xlsx_processor.py`** - Excel processing
  - Sheet data extraction
  - Table structure analysis

- **`utils/markdown_formatter.py`** - Output formatting
  - Converts API results to markdown
  - Handles nested JSON structures
  - Manages file output

### Agent Tools

The LangChain agent provides six main tools:

1. **Document Processing** - Extract structured content from various file formats
2. **File Reading** - Read markdown files line by line with range support
3. **File Editing** - Edit markdown files at specific lines (replace/insert/append)
4. **Content Search** - Search for terms in markdown files with case sensitivity options
5. **File Listing** - List and analyze markdown files in directories
6. **Triple Extraction** - Extract knowledge graph triples from markdown content using OpenIE

### Data Flow

```
User Input → CLI → Main Module
                ↓
        [Document Processing] or [Agent Mode] or [Workflow Mode]
                ↓
    Format-Specific Processors → API Integration
                ↓
        Content Extraction & Analysis
                ↓
    [Workflow Mode Only]
        ↓
    LLM-based Semantic Chunking → Topic Boundary Detection
                ↓
    OpenIE Triple Extraction → Knowledge Graph Construction
                ↓
    Output Formatting (Text/JSON/Markdown/Chunks/Triples)
```

## Examples

### Complete Workflow Example

```bash
# Process a financial report and extract knowledge graph triples
uv run kg-extractor workflow financial_report.pdf

# Output files:
# - output/financial_report_analysis.md (structured content)
# - output/financial_report_chunks.json (semantic chunks)
# - output/financial_report_triples.json (knowledge graph triples)

# The triples file contains:
# - Subject-predicate-object relationships
# - Two-level predicate hierarchy
# - Causal links with confidence weights
# - Temporal metadata
# - Evidence quotes
```

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

### Workflow Mode Examples

```bash
# Complete workflow with default settings
uv run kg-extractor workflow document.pdf

# Workflow with custom chunking granularity
uv run kg-extractor workflow document.pdf --similarity-threshold 0.3 --min-chunk-size 50 --max-chunk-size 800

# Workflow with different providers for each step
uv run kg-extractor workflow document.pdf --provider nvidia --chunking-llm-provider openai --triplet-llm-provider openai

# Workflow focused on diagrams
uv run kg-extractor workflow presentation.pptx --content-type diagram --similarity-threshold 0.4

# Workflow with specific models
uv run kg-extractor workflow report.docx --model microsoft/phi-4-multimodal-instruct --chunking-llm-model gpt-4o-mini --triplet-llm-model gpt-4o-mini
```

**Workflow Output:**

The workflow generates three output files:

1. **Markdown Analysis** (`output/<filename>_analysis.md`)
   - Structured content extracted from the document
   - Text, diagrams, and tables in markdown format
   - Content filtering applied (skips TOC, references, etc.)

2. **Semantic Chunks** (`output/<filename>_chunks.json`)
   - Semantically coherent chunks based on topic shifts
   - Each chunk contains chunk_id and content
   - LLM-based topic boundary detection

3. **Knowledge Graph Triples** (`output/<filename>_triples.json`)
   - Subject-predicate-object triples with OpenIE
   - Two-level predicate hierarchy (Specific Predicate → Relationship Class)
   - Causal links with triggered_by, mechanism, and causal_weight
   - Temporal metadata (validFrom, validTo, status)
   - Evidence quotes for each triple

### Output File Formats

#### Markdown Analysis Format

The markdown analysis file contains structured content with:

- Document type and page information
- Text content with improved readability
- Diagram descriptions with data insights
- Table data in JSON format
- Metadata about content quality

#### Semantic Chunks Format

```json
{
  "source_file": "document_analysis.md",
  "total_chunks": 5,
  "similarity_threshold": 0.5,
  "min_chunk_size": 100,
  "max_chunk_size": 1000,
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "chunks": [
    {
      "chunk_id": 1,
      "content": "Content of chunk 1..."
    },
    {
      "chunk_id": 2,
      "content": "Content of chunk 2..."
    }
  ]
}
```

#### Knowledge Graph Triples Format

```json
{
  "source_file": "document_triples.json",
  "total_chunks": 5,
  "total_triples": 51,
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "chunks": [
    {
      "chunk_id": 1,
      "triples": [
        {
          "subject": {
            "name": "Entity Name",
            "type": "Organization"
          },
          "predicate": "SPECIFIC_VERB",
          "object": {
            "name": "Entity Name",
            "type": "Concept"
          },
          "properties": {
            "relationship_class": "BROAD_CATEGORY",
            "status": "Current",
            "evidence_quote": "Exact text from source",
            "validFrom": "2024-01-01",
            "validTo": null,
            "causal_link": {
              "triggered_by": "Event",
              "mechanism": "How it happened",
              "causal_weight": 0.8
            }
          }
        }
      ]
    }
  ],
  "all_triples": [...]
}
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

#### Workflow Mode

- `--provider`: API provider for document processing - nvidia/openrouter (default: nvidia)
- `--model`: Model for document processing (default: microsoft/phi-4-multimodal-instruct)
- `--content-type`: Content focus - text/diagram/table/mixed (default: mixed)
- `--chunking-llm-provider`: LLM provider for chunking - openai/groq/nvidia/openrouter (default: openai)
- `--chunking-llm-model`: Model for chunking (default: gpt-4o-mini)
- `--similarity-threshold`: Threshold for topic change detection (0.0-1.0, default: 0.5)
- `--min-chunk-size`: Minimum tokens per chunk (default: 100)
- `--max-chunk-size`: Maximum tokens per chunk (default: 1000)
- `--triplet-llm-provider`: LLM provider for triple extraction - openai/groq/nvidia/openrouter (default: openai)
- `--triplet-llm-model`: Model for triple extraction (default: gpt-4o-mini)

### Recommended Models

#### OpenAI (Best for Agent Mode, Chunking, and Triple Extraction)
- `gpt-4o-mini` - Fast, cost-effective, excellent tool calling and reasoning
- `gpt-4o` - Higher quality, more expensive, better for complex analysis

#### NVIDIA (Good for Vision Tasks - Document Processing)
- `microsoft/phi-4-multimodal-instruct` - Excellent for document analysis
- `google/gemma-3-27b-it` - Good general purpose model

#### OpenRouter (Free Options)
- `openai/gpt-oss-120b:free` - Good tool calling support
- `google/gemma-4-31b-it:free` - May have rate limits

#### Groq (Fast Inference)
- `openai/gpt-oss-120b` - Very fast inference with Groq

**Model Selection for Workflow:**

- **Document Processing**: Use NVIDIA models (vision capabilities)
- **Semantic Chunking**: Use OpenAI models (better comprehension)
- **Triple Extraction**: Use OpenAI models (better reasoning and extraction)

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

# Test agent with triple extraction
uv run kg-extractor --agent --agent-task "Extract triples from output/test_analysis.md"
```

### Testing Workflow Functionality

```bash
# Test workflow with default settings
uv run kg-extractor workflow test.pdf

# Test workflow with custom chunking
uv run kg-extractor workflow test.pdf --similarity-threshold 0.3 --min-chunk-size 50

# Test workflow with different providers
uv run kg-extractor workflow test.pdf --provider nvidia --chunking-llm-provider openai --triplet-llm-provider openai

# Verify output files
ls output/test_analysis.md
ls output/test_analysis_chunks.json
ls output/test_triples.json
```

### Module Development

When adding new functionality:

1. **Create format-specific processor** (for new file types)
   - Follow the pattern in `processors/pdf_processor.py`
   - Implement `process_<format>()` function
   - Add to `utils/input_processor.py`

2. **Add new agent tools** (for new capabilities)
   - Create tool function with `@tool` decorator in `tools/agent.py`
   - Add to tools list in `tools/agent.py`
   - Update system prompt if needed

3. **Add new prompts** (for different content types)
   - Add to `utils/prompts.py`
   - Update `get_content_specific_prompt()` function
   - Follow existing prompt structure for consistency

4. **Create new utility modules** (for new functionality)
   - Add to `utils/` directory
   - Follow existing patterns (e.g., `semantic_chunker.py`, `triple_extractor.py`)
   - Include proper error handling and logging

5. **Update workflow** (for new processing steps)
   - Modify `tools/workflow.py` to include new steps
   - Update CLI arguments in `main.py`
   - Ensure parallel processing is used where appropriate

6. **Update tests**
   - Add tests for new functionality
   - Ensure existing tests still pass
   - Update test fixtures if needed

## Performance Considerations

### Document Processing

- **Large PDFs**: Multi-page documents may take significant time to process
- **High-resolution images**: Larger images require more API tokens and processing time
- **Complex layouts**: Documents with complex formatting may need longer processing
- **Parallel processing**: Multi-page documents are processed in parallel (max_workers=5) for improved performance

### Semantic Chunking

- **LLM analysis time**: Chunking requires LLM analysis of the full content
- **Content size**: Larger documents may take longer to analyze and chunk
- **Granularity settings**: Lower similarity thresholds may result in more chunks and longer processing time

### Triple Extraction

- **Parallel processing**: Chunks are processed in parallel (max_workers=5) for efficient extraction
- **Chunk count**: More chunks mean more parallel processing but also more API calls
- **Complexity**: Documents with many entities and relationships will take longer to process
- **Causal discovery**: Identifying causal links adds processing time but provides richer insights

### API Usage

- **Token limits**: Be mindful of token limits when processing large documents
- **Rate limiting**: Some providers have rate limits, especially free tiers
- **Cost optimization**: Use appropriate models for different tasks
  - Use faster/cheaper models for simple tasks
  - Use higher-quality models for complex analysis
- **Workflow mode**: Uses multiple API calls (document processing + chunking + triple extraction)
  - Consider using different providers for each step to optimize cost and performance
  - NVIDIA for vision tasks, OpenAI for chunking and triple extraction

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

### Workflow Mode

1. **Choose appropriate granularity**
   - Use `--similarity-threshold 0.3-0.5` for detailed analysis
   - Use `--similarity-threshold 0.5-0.7` for balanced chunking
   - Use `--similarity-threshold 0.7-1.0` for high-level overview

2. **Optimize chunk sizes**
   - Adjust `--min-chunk-size` and `--max-chunk-size` based on content
   - Smaller chunks for detailed analysis (50-500 tokens)
   - Larger chunks for high-level overview (500-1500 tokens)

3. **Select appropriate providers**
   - Use NVIDIA for document processing (vision models)
   - Use OpenAI for chunking and triple extraction (better tool support)
   - Consider cost vs. quality trade-offs

4. **Review output quality**
   - Check chunks for semantic coherence
   - Verify triples for accuracy and completeness
   - Review causal links for plausibility

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

4. **Leverage triple extraction**
   - Use extract_triples_tool for knowledge graph construction
   - Combine with search and read tools for comprehensive analysis
   - Review extracted triples for accuracy and completeness

## Future Enhancements

Potential areas for expansion:

- **Additional file formats**: Support for more document types (EPUB, HTML, etc.)
- **Advanced chunking strategies**: Hybrid approaches combining LLM and embeddings
- **Knowledge graph visualization**: Interactive graph visualization tools
- **Entity resolution**: Cross-document entity linking and disambiguation
- **Relationship inference**: Discover implicit relationships between entities
- **Graph database integration**: Neo4j, ArangoDB, or other graph databases
- **Batch processing**: Process multiple files in parallel with workflow mode
- **Custom prompts**: User-defined prompts for specific use cases
- **Output templates**: Customizable output formats and templates
- **Performance optimization**: Caching and incremental processing
- **Collaboration features**: Multi-user support and sharing
- **API endpoints**: REST API for programmatic access
- **Web interface**: Browser-based UI for document processing

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

### Workflow Mode Issues

**Chunking produces too many/few chunks:**
- Adjust `--similarity-threshold` (lower = more chunks, higher = fewer chunks)
- Check that the content is appropriate for semantic chunking
- Review chunk sizes with `--min-chunk-size` and `--max-chunk-size`

**Triple extraction returns few or no triples:**
- Check that chunks contain meaningful content
- Try with a different model for triple extraction
- Review the prompt for clarity and specificity
- Ensure the content has extractable relationships

**Causal links not discovered:**
- Causal links are only extracted when explicitly mentioned in the text
- Review the content for causal language (e.g., "due to", "because of", "following")
- Check that the model supports complex reasoning

**Workflow takes too long:**
- Large documents with many chunks will take longer to process
- Consider using faster models for chunking and triple extraction
- Reduce the number of chunks by increasing similarity threshold
- Check API rate limits and consider using different providers

## License

Add your license here.