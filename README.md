# KG Extractor

Extract structured content from documents using AI APIs for knowledge graph construction, with intelligent agent-based processing, LLM-based semantic chunking, and OpenIE triple extraction with causal discovery.

## Features

- Multi-API provider support: OpenAI, NVIDIA, OpenRouter, Groq
- Multi-format support: Images (PNG, JPEG, WEBP, GIF, BMP), Documents (PDF, DOCX, PPTX, XLSX)
- Three-step workflow: Document processing → LLM-based semantic chunking → OpenIE triple extraction
- Intelligent content processing: Text restructuring, diagram analysis, table-to-JSON conversion
- Semantic chunking: LLM-based topic boundary detection (not embedding-based)
- Knowledge graph triple extraction: OpenIE with zero-shot discovery, causal discovery, temporal metadata
- LangChain Agent Integration: Document processing, markdown editing, triple extraction tools
- Modular architecture with comprehensive error handling

## Key Technical Concepts

### OpenIE (Open Information Extraction)
Zero-shot discovery of predicates and categories without predefined ontology. Two-level hierarchy: Specific Predicate → Relationship Class.

### LLM-based Semantic Chunking
Uses LLM comprehension to detect topic shifts (not embedding-based). Configurable granularity via similarity threshold (0.0-1.0).

### Causal Discovery
Identifies causal relationships with trigger identification, mechanism extraction, and confidence scoring (0.0-1.0).

### Parallel Processing
Multi-page documents and chunks processed in parallel (max_workers=5) for efficiency.

## Use Cases

- **Knowledge Graph Construction**: Extract entities and relationships from research papers, business documents, technical documentation
- **Document Analysis**: Extract key findings, identify causal relationships, summarize technical documentation
- **Content Organization**: Create semantically coherent sections from long documents, identify topic boundaries
- **Information Extraction**: Extract entities, relationships, temporal information, and causal mechanisms

## Project Structure

```
kg-extractor/
├── src/kg_extractor/
│   ├── main.py              # CLI entry point
│   ├── tools/               # Agent tools and workflow orchestration
│   │   ├── agent.py         # LangChain agent with tools
│   │   └── workflow.py      # Three-step workflow orchestrator
│   ├── utils/               # Utility modules
│   │   ├── parser.py        # API integration
│   │   ├── prompts.py       # System and content prompts
│   │   ├── markdown_formatter.py # Markdown output formatting
│   │   ├── input_processor.py # Main document processor
│   │   ├── semantic_chunker.py # LLM-based semantic chunking
│   │   └── triple_extractor.py # OpenIE triple extraction
│   └── processors/          # Format-specific processors
│       ├── image_processor.py
│       ├── pdf_processor.py
│       ├── docx_processor.py
│       ├── pptx_processor.py
│       └── xlsx_processor.py
├── tests/
├── output/                  # Generated markdown, chunks, and triples files
├── pyproject.toml
└── .env.example
```

## Getting Started

### Prerequisites

- Python 3.10+
- `uv` installed (https://github.com/astral-sh/uv)
- API keys for one or more providers: OpenAI, NVIDIA, OpenRouter, Groq

### Installation

```bash
# Install dependencies
uv sync

# Install in development mode
uv pip install -e .
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_openai_api_key_here
# NVIDIA_API_KEY=your_nvidia_api_key_here
# OPENROUTER_API_KEY=your_openrouter_api_key_here
# GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Quick Start

```bash
# Run complete workflow (recommended)
uv run kg-extractor workflow document.pdf
```

### Document Processing Mode

```bash
# Basic usage
uv run kg-extractor document.pdf
uv run kg-extractor image.jpg
uv run kg-extractor report.docx

# Options
uv run kg-extractor --stream image.png
uv run kg-extractor --content-type table spreadsheet.xlsx
uv run kg-extractor --format json --output result.json report.pdf
uv run kg-extractor --provider openai --model gpt-4o-mini document.pdf
uv run kg-extractor --max-tokens 4096 --temperature 0.3 document.pdf
```

### Agent Mode

```bash
# Interactive agent mode
uv run kg-extractor --agent

# Single task mode
uv run kg-extractor --agent --agent-task "List all markdown files in the output directory"
uv run kg-extractor --agent --agent-task "Process document.pdf and save as markdown"
uv run kg-extractor --agent --agent-task "Search for 'EBITDA' in output/test_analysis.md"

# Use specific model
uv run kg-extractor --agent --agent-model gpt-4o-mini --agent-task "List all markdown files"
```

**Available Agent Tools:**
- `process_document_tool` - Process documents and extract structured content
- `read_markdown_file` - Read markdown files line by line
- `edit_markdown_file` - Edit markdown files at specific lines
- `search_markdown_content` - Search for content in markdown files
- `list_markdown_files` - List all markdown files in a directory
- `extract_triples_tool` - Extract knowledge graph triples from markdown content

### Workflow Mode

```bash
# Run complete workflow
uv run kg-extractor workflow document.pdf

# Customize providers and models
uv run kg-extractor workflow document.pdf --provider nvidia --chunking-llm-provider openai --triplet-llm-provider openai

# Customize chunking parameters
uv run kg-extractor workflow document.pdf --similarity-threshold 0.3 --min-chunk-size 50 --max-chunk-size 800

# Focus on specific content types
uv run kg-extractor workflow document.pdf --content-type diagram
```

**Workflow Steps:**
1. Document Processing - Extract structured content using AI vision models
2. Semantic Chunking - LLM-based chunking that detects topic shifts
3. Triple Extraction - OpenIE extraction with zero-shot ontology discovery

**Output Files:**
- `output/<filename>_analysis.md` - Structured markdown from document processing
- `output/<filename>_chunks.json` - Semantically chunked content
- `output/<filename>_triples.json` - Knowledge graph triples with causal links

### Supported Formats

**Images:** PNG, JPEG, WEBP, GIF, BMP
**Documents:** PDF, DOCX, PPTX, XLSX

### Content Processing Features

**Text Processing:** Extracts and restructures text, fixes broken sentences using context, maintains technical accuracy.

**Diagram & Chart Analysis:** Provides detailed explanations of visual elements, identifies chart types, explains data relationships and patterns.

**Table Processing:** Converts tables to structured JSON format, preserves data types and structure, handles merged cells.

**Smart Content Filtering:** Skips title pages, table of contents, references, appendices. Focuses on main content, methodology, results, conclusions.

**Semantic Chunking:** LLM-based topic detection, configurable granularity (0.0-1.0), semantic coherence preservation, topic boundary detection.

**Knowledge Graph Triple Extraction:** Zero-shot ontology discovery, two-level predicate hierarchy, causal discovery with confidence weights, temporal metadata, evidence-based extraction, parallel processing.

**Triple Structure:**
```json
{
  "subject": {"name": "Entity Name", "type": "Person/Organization/Concept"},
  "predicate": "SPECIFIC_VERB_IN_ALL_CAPS",
  "object": {"name": "Entity Name", "type": "Person/Organization/Concept"},
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

### Core Modules

- **`tools/agent.py`** - LangChain agent integration with tool-based architecture
- **`tools/workflow.py`** - Three-step workflow orchestrator with parallel processing
- **`utils/parser.py`** - API integration layer for NVIDIA, OpenRouter, Groq, OpenAI
- **`utils/semantic_chunker.py`** - LLM-based semantic chunking with topic boundary detection
- **`utils/triple_extractor.py`** - OpenIE triple extraction with zero-shot ontology discovery
- **`utils/input_processor.py`** - Main document processor with format-specific delegation
- **`utils/prompts.py`** - System and content-specific prompts
- **`processors/`** - Format-specific processors (image, PDF, DOCX, PPTX, XLSX)
- **`utils/markdown_formatter.py`** - Output formatting for markdown and JSON

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

### Complete Workflow

```bash
# Process a financial report and extract knowledge graph triples
uv run kg-extractor workflow financial_report.pdf

# Output files:
# - output/financial_report_analysis.md (structured content)
# - output/financial_report_chunks.json (semantic chunks)
# - output/financial_report_triples.json (knowledge graph triples)
```

### Document Analysis

```bash
# Process a PDF document and save as markdown
uv run kg-extractor --format markdown report.pdf
```

### Agent-Based Analysis

```bash
# Interactive session to analyze multiple documents
uv run kg-extractor --agent

# In the agent session, you can:
# - Process documents: "Process document.pdf and save as markdown"
# - Analyze content: "Read output/document_analysis.md and summarize the key findings"
# - Search for terms: "Search for 'revenue' in all markdown files"
# - Edit files: "Edit line 10 of output/summary.md to add more details"
```

### Advanced Workflows

```bash
# Complex multi-step analysis
uv run kg-extractor --agent --agent-task "Process financial_report.pdf, then search the output for 'EBITDA', and finally create a summary of the financial metrics found"

# Document comparison
uv run kg-extractor --agent --agent-task "Read the first 20 lines of output/report1.md and output/report2.md, then compare their key differences"

# Workflow with custom settings
uv run kg-extractor workflow document.pdf --similarity-threshold 0.3 --min-chunk-size 50 --max-chunk-size 800
```

### Output File Formats

**Markdown Analysis:** Structured content with document type, text, diagrams, tables, and metadata.

**Semantic Chunks:**
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
    {"chunk_id": 1, "content": "Content of chunk 1..."},
    {"chunk_id": 2, "content": "Content of chunk 2..."}
  ]
}
```

**Knowledge Graph Triples:**
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
          "subject": {"name": "Entity Name", "type": "Organization"},
          "predicate": "SPECIFIC_VERB",
          "object": {"name": "Entity Name", "type": "Concept"},
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

## API Parameters

### Document Processing Mode

- `--provider`: API provider - nvidia/openrouter (default: nvidia)
- `--model`: Model to use (default varies by provider)
- `--max-tokens`: Maximum tokens in response (default: 2048)
- `--temperature`: Sampling temperature, 0-2 (default: 0.20)
- `--top-p`: Nucleus sampling parameter, 0-1 (default: 0.70)
- `--content-type`: Content focus - text/diagram/table/mixed (default: mixed)
- `--format`: Output format - text/json/markdown (default: text)
- `--stream`: Use streaming API response (images only)
- `--output`: Save output to file instead of printing

### Agent Mode

- `--agent`: Enable agent mode
- `--agent-model`: Model for agent (default: gpt-4o-mini)
- `--agent-task`: Single task for agent (non-interactive mode)

### Workflow Mode

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

## Recommended Models

**OpenAI** (Best for Agent Mode, Chunking, and Triple Extraction)
- `gpt-4o-mini` - Fast, cost-effective, excellent tool calling and reasoning
- `gpt-4o` - Higher quality, more expensive, better for complex analysis

**NVIDIA** (Good for Vision Tasks - Document Processing)
- `microsoft/phi-4-multimodal-instruct` - Excellent for document analysis
- `google/gemma-3-27b-it` - Good general purpose model

**OpenRouter** (Free Options)
- `openai/gpt-oss-120b:free` - Good tool calling support
- `google/gemma-4-31b-it:free` - May have rate limits

**Groq** (Fast Inference)
- `openai/gpt-oss-120b` - Very fast inference with Groq

**Model Selection for Workflow:**
- Document Processing: Use NVIDIA models (vision capabilities)
- Semantic Chunking: Use OpenAI models (better comprehension)
- Triple Extraction: Use OpenAI models (better reasoning and extraction)

## Development

```bash
# Add dependencies
uv add package-name              # Runtime dependency
uv add --dev package-name        # Development dependency

# Run tests
uv run pytest                    # All tests
uv run pytest tests/test_main.py # Specific file
uv run pytest -v                 # Verbose output
uv run pytest --cov=kg_extractor # With coverage

# Code quality
uv run black src/ tests/         # Format code
uv run ruff check src/ tests/   # Lint code
uv run mypy src/                 # Type check
uv run ruff check --fix src/ tests/ # Fix linting issues
```

### Testing

```bash
# Test agent functionality
uv run kg-extractor --agent --agent-task "Say hello"
uv run kg-extractor --agent --agent-task "List all markdown files in the output directory"

# Test workflow functionality
uv run kg-extractor workflow test.pdf
uv run kg-extractor workflow test.pdf --similarity-threshold 0.3 --min-chunk-size 50
```

### Module Development

1. **Format-specific processors**: Follow pattern in `processors/pdf_processor.py`, add to `utils/input_processor.py`
2. **Agent tools**: Create with `@tool` decorator in `tools/agent.py`, add to tools list
3. **Prompts**: Add to `utils/prompts.py`, update `get_content_specific_prompt()`
4. **Utility modules**: Add to `utils/`, follow existing patterns with error handling
5. **Workflow**: Modify `tools/workflow.py`, update CLI arguments in `main.py`
6. **Tests**: Add tests for new functionality, ensure existing tests pass

## Performance Considerations

**Document Processing:** Large PDFs and high-resolution images take significant time. Multi-page documents processed in parallel (max_workers=5).

**Semantic Chunking:** Requires LLM analysis of full content. Lower similarity thresholds result in more chunks and longer processing.

**Triple Extraction:** Chunks processed in parallel (max_workers=5). More chunks = more API calls. Causal discovery adds processing time.

**API Usage:** Be mindful of token limits and rate limits. Use faster/cheaper models for simple tasks, higher-quality models for complex analysis. Workflow mode uses multiple API calls.

**Agent Mode:** Tool calling adds latency. Complex tasks require multiple API calls. Long conversations may hit context limits.

## Best Practices

**Document Processing:**
- Use `--content-type table` for spreadsheets, `--content-type diagram` for presentations with charts, `--content-type text` for text-heavy documents
- Use `--format markdown` for readable results, `--format json` for programmatic processing
- Adjust `--max-tokens` based on document complexity

**Workflow Mode:**
- Use `--similarity-threshold 0.3-0.5` for detailed analysis, `0.5-0.7` for balanced chunking, `0.7-1.0` for high-level overview
- Adjust `--min-chunk-size` and `--max-chunk-size` based on content (50-500 tokens for detailed, 500-1500 for overview)
- Use NVIDIA for document processing, OpenAI for chunking and triple extraction
- Review chunks for semantic coherence, triples for accuracy, causal links for plausibility

**Agent Usage:**
- Be specific with tasks, break complex tasks into smaller steps
- Use OpenAI models for best tool support
- Leverage triple extraction for knowledge graph construction

## Future Enhancements

- Additional file formats (EPUB, HTML, etc.)
- Advanced chunking strategies (hybrid LLM and embeddings)
- Knowledge graph visualization tools
- Entity resolution and cross-document linking
- Graph database integration (Neo4j, ArangoDB)
- Batch processing for multiple files
- Custom prompts and output templates
- Performance optimization with caching
- REST API and web interface

## Troubleshooting

**API Key Issues:** Ensure `.env` file exists in project root with appropriate API key (OPENAI_API_KEY, NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY).

**Agent Mode Issues:** Use OpenAI models (gpt-4o-mini) for best tool support. Free models may have rate limits.

**File Processing Errors:** Check file path, format support, and read permissions. Password-protected files not supported.

**API Errors:** Verify API key, check rate limits, ensure internet connection, verify model name.

**Content Quality Issues:** Increase `--max-tokens` for incomplete extraction, ensure source documents are clear and readable.

**Workflow Mode Issues:** Adjust `--similarity-threshold` for chunking (lower = more chunks). Ensure chunks contain meaningful content for triple extraction. Causal links only extracted when explicitly mentioned.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.