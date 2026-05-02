# KG Extractor

Extract structured content from documents using AI APIs for knowledge graph construction, with intelligent agent-based processing, LLM-based semantic chunking, and OpenIE triple extraction with causal discovery.

## Features

- Multi-API provider support: OpenAI, NVIDIA, OpenRouter, Groq
- Multi-format support: Images (PNG, JPEG, WEBP, GIF, BMP), Documents (PDF, DOCX, PPTX, XLSX)
- LangGraph workflow: Document processing → LLM-based semantic chunking → OpenIE triple extraction
- Intelligent content processing: Text restructuring, diagram analysis, table-to-JSON conversion
- Semantic chunking: LLM-based topic boundary detection (not embedding-based)
- Knowledge graph triple extraction: OpenIE with zero-shot discovery, causal discovery, temporal metadata
- LangChain Agent Integration: Document processing, markdown editing, triple extraction tools
- Triple refinement with Qdrant: Entity resolution and canonical ID matching
- Neo4j graph building: Automatic knowledge graph construction from refined triples
- Schema validation: Validate triples and graph against predefined schema (optional)
- Modular architecture with comprehensive error handling

## Key Technical Concepts

### OpenIE (Open Information Extraction)
Zero-shot discovery of predicates and categories without predefined ontology. Two-level hierarchy: Specific Predicate → Relationship Class.

### LLM-based Semantic Chunking
Uses LLM comprehension to detect topic shifts (not embedding-based). Configurable granularity via similarity threshold (0.0-1.0).

### Causal Discovery
Identifies causal relationships with trigger identification, mechanism extraction, and confidence scoring (0.0-1.0).

### Schema Validation
Optional validation against predefined schema to ensure extracted triples conform to expected node types, relations, and enum values. Invalid entities and relationships are rejected during graph building.

### Parallel Processing
Multi-page documents and chunks processed in parallel (max_workers=5) for efficiency.

## Use Cases

- **Knowledge Graph Construction**: Extract entities and relationships from research papers, business documents, technical documentation
- **Document Analysis**: Extract key findings, identify causal relationships, summarize technical documentation
- **Content Organization**: Create semantically coherent sections from long documents, identify topic boundaries
- **Information Extraction**: Extract entities, relationships, temporal information, and causal mechanisms
- **Entity Resolution**: Use Qdrant to resolve duplicate entities and assign canonical IDs
- **Graph Database Integration**: Build Neo4j knowledge graphs from refined triples for advanced querying
- **Schema-Compliant Extraction**: Ensure extracted knowledge graphs conform to predefined schemas for domain-specific applications

## Project Structure

```
kg-extractor/
├── src/kg_extractor/
│   ├── main.py              # CLI entry point
│   ├── tools/               # Agent tools and workflow orchestration
│   │   ├── agent.py         # LangChain agent with tools
│   │   └── langgraph_workflow.py # LangGraph workflow with state management
│   ├── utils/               # Utility modules
│   │   ├── parser.py        # API integration
│   │   ├── prompts.py       # System and content prompts
│   │   ├── markdown_formatter.py # Markdown output formatting
│   │   ├── input_processor.py # Main document processor
│   │   ├── semantic_chunker.py # LLM-based semantic chunking
│   │   ├── triple_extractor.py # OpenIE triple extraction
│   │   ├── triple_refiner.py # Qdrant-based triple refinement
│   │   ├── neo4j_graph_builder.py # Neo4j graph construction
│   │   ├── schema_parser.py # Schema validation and parsing
│   │   └── schema.md        # Schema definition for validation
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

# For triple refinement with Qdrant (optional)
# QDRANT_URL=your_qdrant_url_here
# QDRANT_API_KEY=your_qdrant_api_key_here

# For Neo4j graph building (optional)
# NEO4J_URI=your_neo4j_uri_here
# NEO4J_USER=your_neo4j_username_here
# NEO4J_PASSWORD=your_neo4j_password_here
```

## Usage

### Quick Start

```bash
# Run complete LangGraph workflow (recommended)
uv run kg-extractor langgraph document.pdf
```

### Document Processing Mode

```bash
# Basic usage (backward compatible - can omit 'process' subcommand)
uv run kg-extractor process document.pdf
uv run kg-extractor image.jpg
uv run kg-extractor report.docx

# Options
uv run kg-extractor process --stream image.png
uv run kg-extractor process --content-type table spreadsheet.xlsx
uv run kg-extractor process --format json --output result.json report.pdf
uv run kg-extractor process --provider openai --model gpt-4o document.pdf
uv run kg-extractor process --max-tokens 4096 --temperature 0.3 document.pdf
```

**Supported Providers:**
- `nvidia` - NVIDIA API (default, good for vision tasks)
- `openrouter` - OpenRouter API (free options available)
- `openai` - OpenAI API (high quality, gpt-4o recommended)
- `groq` - Groq API (fast inference)
- `google` - Google Gemini API (gemini-3.1-flash-lite-preview)

**Recommended Models:**
- NVIDIA: `microsoft/phi-4-multimodal-instruct` (vision), `google/gemma-3-27b-it` (general)
- OpenRouter: `google/gemma-4-31b-it:free` (free), `openai/gpt-oss-120b:free` (free)
- OpenAI: `gpt-4o` (best quality), `gpt-4o-mini` (cost-effective)
- Groq: `openai/gpt-oss-120b` (very fast)
- Google: `google/gemini-3.1-flash-lite-preview` (fast, cost-effective)

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

### LangGraph Workflow Mode

```bash
# Run complete LangGraph workflow
uv run kg-extractor langgraph document.pdf

# Customize providers and models
uv run kg-extractor langgraph document.pdf --provider openai --chunking-llm-provider openai --triplet-llm-provider openai

# Customize chunking parameters
uv run kg-extractor langgraph document.pdf --similarity-threshold 0.3 --min-chunk-size 50 --max-chunk-size 800

# Focus on specific content types
uv run kg-extractor langgraph document.pdf --content-type diagram

# Enable schema validation
uv run kg-extractor langgraph document.pdf --with-schema
```

**Supported Providers for Document Processing:**
- `nvidia` - NVIDIA API (default, good for vision tasks)
- `openrouter` - OpenRouter API (free options available)
- `openai` - OpenAI API (high quality, gpt-4o recommended)

**LangGraph Workflow Steps:**
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

**Triple Refinement with Qdrant:** Entity resolution using vector similarity, canonical ID assignment, duplicate detection and merging, confidence scoring.

**Neo4j Graph Building:** Automatic node and relationship creation, property preservation, constraint handling, batch processing for performance.

**Schema Validation:** Optional validation against predefined schema, ensures node types and relations conform to expected definitions, validates enum values, rejects invalid entities during graph building.

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

**Refined Triple Structure (with Qdrant):**
```json
{
  "subject": {
    "name": "Entity Name",
    "type": "Person/Organization/Concept",
    "canonical_id": "entity_12345",
    "confidence": 0.95
  },
  "predicate": "SPECIFIC_VERB_IN_ALL_CAPS",
  "object": {
    "name": "Entity Name",
    "type": "Person/Organization/Concept",
    "canonical_id": "entity_67890",
    "confidence": 0.92
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

### Core Modules

- **`tools/agent.py`** - LangChain agent integration with tool-based architecture
- **`tools/langgraph_workflow.py`** - LangGraph workflow with state management
- **`utils/parser.py`** - API integration layer for NVIDIA, OpenRouter, Groq, OpenAI
- **`utils/semantic_chunker.py`** - LLM-based semantic chunking with topic boundary detection
- **`utils/triple_extractor.py`** - OpenIE triple extraction with zero-shot ontology discovery
- **`utils/triple_refiner.py`** - Qdrant-based triple refinement for entity resolution
- **`utils/neo4j_graph_builder.py`** - Neo4j graph construction from refined triples
- **`utils/schema_parser.py`** - Schema validation and parsing for knowledge graph compliance
- **`utils/input_processor.py`** - Main document processor with format-specific delegation
- **`utils/prompts.py`** - System and content-specific prompts
- **`processors/`** - Format-specific processors (image, PDF, DOCX, PPTX, XLSX)
- **`utils/markdown_formatter.py`** - Output formatting for markdown and JSON

### Data Flow

```
User Input → CLI → Main Module
                ↓
        [Document Processing] or [Agent Mode] or [LangGraph Mode]
                ↓
    Format-Specific Processors → API Integration
                ↓
        Content Extraction & Analysis
                ↓
    [LangGraph Mode Only]
        ↓
    LLM-based Semantic Chunking → Topic Boundary Detection
                ↓
    OpenIE Triple Extraction → Knowledge Graph Construction
                ↓
    [Optional: Schema Validation] → Validate against schema.md
                ↓
    [Optional: Triple Refinement with Qdrant]
                ↓
    [Optional: Neo4j Graph Building with Schema Validation]
                ↓
    Output Formatting (Text/JSON/Markdown/Chunks/Triples/Refined Triples)
```

## Examples

### Complete LangGraph Workflow

```bash
# Process a financial report and extract knowledge graph triples
uv run kg-extractor langgraph financial_report.pdf

# Output files:
# - output/financial_report_analysis.md (structured content)
# - output/financial_report_chunks.json (semantic chunks)
# - output/financial_report_triples.json (knowledge graph triples)
# - output/financial_report_triples_refined.json (refined triples with Qdrant)
# - Neo4j graph database (if build-graph enabled)
```

### Complete Pipeline with Refinement

```bash
# Step 1: Run LangGraph workflow with refinement and graph building
uv run kg-extractor langgraph document.pdf --refine-triples --build-graph

# Step 2: Or run steps separately
# Extract triples
uv run kg-extractor langgraph document.pdf --no-refine-triples --no-build-graph

# Refine triples with Qdrant
uv run kg-extractor refine output/document_triples.json

# Build Neo4j graph
uv run kg-extractor graph output/document_triples_refined.json
```

### Document Analysis

```bash
# Process a PDF document and save as markdown
uv run kg-extractor process --format markdown report.pdf
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

# LangGraph workflow with custom settings
uv run kg-extractor langgraph document.pdf --similarity-threshold 0.3 --min-chunk-size 50 --max-chunk-size 800

# LangGraph workflow with refinement and graph building
uv run kg-extractor langgraph document.pdf --refine-triples --build-graph

# LangGraph workflow with schema validation
uv run kg-extractor langgraph document.pdf --with-schema

# Refine triples with Qdrant for entity resolution
uv run kg-extractor refine output/document_triples.json --llm-provider openai --llm-model gpt-4o-mini

# Build Neo4j graph from refined triples
uv run kg-extractor graph output/document_triples_refined.json
```

### KG Reasoning Mode

```bash
# Query the knowledge graph
uv run kg-reasoning query "What is the relationship between Nvidia and AI?"

# Query with verbose output
uv run kg-reasoning query "How does debt affect company performance?" --verbose

# Query with custom LLM settings
uv run kg-reasoning query "What are the main causes of market volatility?" --llm-provider openai --llm-model gpt-4o-mini

# Query with similarity threshold
uv run kg-reasoning query "What factors influence EBITDA multiples?" --similarity-threshold 0.8

# Save query results to file
uv run kg-reasoning query "What is the impact of M&A advisory services?" --output result.json

# Query using LangGraph workflow
uv run kg-reasoning langgraph "What are the key drivers of business valuation?"
```

**KG Reasoning Workflow:**
1. Entity Extraction - Extract entities from natural language query
2. Entity Matching - Match entities in Qdrant vector database
3. Query Refinement - Refine query with canonical entities
4. Cypher Generation - Generate Neo4j Cypher query
5. Graph Query Execution - Execute query against Neo4j
6. Answer Synthesis - Synthesize natural language answer from results

**Output Files:**
- Query results displayed in terminal or saved to JSON file
- Includes answer, metadata, and query suggestions if no matches found

### Output File Formats

**Markdown Analysis:** Structured content with document type, text, diagrams, tables, and metadata.

**Semantic Chunks:**
```json
{
  "source_file": "document_analysis.md",
  "total_chunks": 5,
  "similarity_threshold": 0.5,
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

**Refined Triples (with Qdrant):**
```json
{
  "source_file": "document_triples_refined.json",
  "total_chunks": 5,
  "total_triples": 51,
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "refinement_stats": {
    "entities_resolved": 15,
    "canonical_ids_assigned": 12,
    "duplicates_merged": 3
  },
  "chunks": [...]
}
```

**Schema Validation:**

The schema is defined in `src/kg_extractor/utils/schema.md` and includes:

- **Node Types**: Tambon, Village, SocialCapital, Domain, Activity, TargetGroup, Impact, Evidence, Resource, EnablingFactor, CommunityIssue, CapabilityDimension, CapabilityAssessment, Innovation, Actor
- **Relations**: BELONGS_TO, LOCATED_IN, PERFORMS, TARGETS, PRODUCES, AFFECTS, SUPPORTS, PARTICIPATES_IN, USES, ENABLED_BY, ADDRESSES, STRENGTHENS, CONNECTED_TO, EMERGES_FROM
- **Enum Values**: SocialCapitalLevel, DomainCode, ScopeLevel, ImpactType, EvidenceStrength, ResourceType

When `--with-schema` is enabled:
- Triple extraction prompts include schema constraints
- Invalid node types are rejected during graph building
- Invalid relations are rejected during graph building
- Enum values are validated against schema definitions
- Only schema-compliant entities and relationships are stored in Neo4j

## API Parameters

### Document Processing Mode

**Command:** `kg-extractor process <file_path>`

- `--provider`: API provider - nvidia/openrouter/openai/groq (default: nvidia)
- `--model`: Model to use (default: google/gemma-3-27b-it for nvidia, gpt-4o for openai)
- `--max-tokens`: Maximum tokens in response (default: 2048)
- `--temperature`: Sampling temperature, 0-2 (default: 0.20)
- `--top-p`: Nucleus sampling parameter, 0-1 (default: 0.70)
- `--content-type`: Content focus - text/diagram/table/mixed (default: mixed)
- `--format`: Output format - text/json/markdown (default: text)
- `--stream`: Use streaming API response (images only)
- `--output`: Save output to file instead of printing

### Agent Mode

**Command:** `kg-extractor --agent`

- `--agent`: Enable agent mode
- `--agent-model`: Model for agent (default: gpt-4o-mini)
- `--agent-task`: Single task for agent (non-interactive mode)

### LangGraph Workflow Mode

**Command:** `kg-extractor langgraph <input_file>`

- `--provider`: API provider for document processing - nvidia/openrouter/openai (default: nvidia)
- `--model`: Model for document processing (default: microsoft/phi-4-multimodal-instruct for nvidia, gpt-4o for openai)
- `--content-type`: Content focus - text/diagram/table/mixed (default: mixed)
- `--chunking-llm-provider`: LLM provider for chunking - openai/groq/nvidia/openrouter (default: openai)
- `--chunking-llm-model`: Model for chunking (default: gpt-4o-mini)
- `--similarity-threshold`: Threshold for topic change detection (0.0-1.0, default: 0.5)
- `--min-chunk-size`: Minimum tokens per chunk (default: 100)
- `--max-chunk-size`: Maximum tokens per chunk (default: 1000)
- `--triplet-llm-provider`: LLM provider for triple extraction - openai/groq/nvidia/openrouter (default: openai)
- `--triplet-llm-model`: Model for triple extraction (default: gpt-4o-mini)
- `--refine-triples` / `--no-refine-triples`: Enable/disable triple refinement (default: True)
- `--refinement-llm-provider`: LLM provider for triple refinement (default: openai)
- `--refinement-llm-model`: Model for triple refinement (default: gpt-4o-mini)
- `--build-graph` / `--no-build-graph`: Enable/disable Neo4j graph building (default: True)
- `--with-schema`: Enable schema validation for triples and graph (default: False)

### Refine Mode

**Command:** `kg-extractor refine <input_file>`

- `--output`: Output file path (default: input_path with _refined suffix)
- `--llm-provider`: LLM provider for canonical comparison (default: openai)
- `--llm-model`: LLM model for canonical comparison (default: gpt-4o-mini)

### Graph Mode

**Command:** `kg-extractor graph <input_file>`

- `--with-schema`: Enable schema validation for graph building (default: False)

Builds Neo4j graph from refined triples. Requires NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables.

```bash
# Build graph from refined triples
uv run kg-extractor graph output/document_triples_refined.json

# Build graph with schema validation
uv run kg-extractor graph output/document_triples_refined.json --with-schema
```

### KG Reasoning Mode

**Command:** `kg-reasoning query <query>`

- `--llm-provider`: LLM provider - openai/groq/nvidia/openrouter (default: openai)
- `--llm-model`: LLM model (default: gpt-4o-mini)
- `--similarity-threshold`: Similarity threshold for entity matching (0.0-1.0, default: 0.75)
- `--output`: Output file path (optional)
- `--verbose`: Show detailed metadata

**Command:** `kg-reasoning langgraph <query>`

Same parameters as query mode, but uses LangGraph for orchestration.

## Recommended Models

**OpenAI** (Best for Agent Mode, Chunking, Triple Extraction, and Document Processing)
- `gpt-4o` - Highest quality, excellent for all tasks including document processing
- `gpt-4o-mini` - Fast, cost-effective, excellent for chunking and triple extraction

**NVIDIA** (Good for Vision Tasks - Document Processing)
- `microsoft/phi-4-multimodal-instruct` - Excellent for document analysis
- `google/gemma-3-27b-it` - Good general purpose model

**OpenRouter** (Free Options)
- `openai/gpt-oss-120b:free` - Good tool calling support
- `google/gemma-4-31b-it:free` - May have rate limits

**Groq** (Fast Inference)
- `openai/gpt-oss-120b` - Very fast inference with Groq

**Model Selection for Workflow:**
- Document Processing: Use OpenAI (gpt-4o) for best quality, or NVIDIA (phi-4-multimodal-instruct) for cost-effectiveness
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

# Test document processing
uv run kg-extractor process test.pdf
uv run kg-extractor process --format markdown test.pdf
```

### Module Development

1. **Format-specific processors**: Follow pattern in `processors/pdf_processor.py`, add to `utils/input_processor.py`
2. **Agent tools**: Create with `@tool` decorator in `tools/agent.py`, add to tools list
3. **Prompts**: Add to `utils/prompts.py`, update `get_content_specific_prompt()`
4. **Utility modules**: Add to `utils/`, follow existing patterns with error handling
5. **LangGraph workflows**: Create new workflows in `tools/langgraph_workflow.py`
6. **Triple refinement**: Extend `utils/triple_refiner.py` for new refinement strategies
7. **Graph builders**: Add new graph database support in `utils/neo4j_graph_builder.py`
8. **Schema validation**: Extend `utils/schema_parser.py` for custom schema definitions
9. **Tests**: Add tests for new functionality, ensure existing tests pass

## Performance Considerations

**Document Processing:** Large PDFs and high-resolution images take significant time. Multi-page documents processed in parallel (max_workers=5).

**Semantic Chunking:** Requires LLM analysis of full content. Lower similarity thresholds result in more chunks and longer processing.

**Triple Extraction:** Chunks processed in parallel (max_workers=5). More chunks = more API calls. Causal discovery adds processing time.

**Triple Refinement:** Qdrant vector search adds latency but improves entity resolution. Batch processing helps reduce API calls.

**Graph Building:** Neo4j batch operations for performance. Large graphs may require memory optimization.

**API Usage:** Be mindful of token limits and rate limits. Use faster/cheaper models for simple tasks, higher-quality models for complex analysis. Workflow mode uses multiple API calls.

**Agent Mode:** Tool calling adds latency. Complex tasks require multiple API calls. Long conversations may hit context limits.

**LangGraph Workflow:** State management adds overhead but provides better control. Suitable for complex multi-step processes.

## Best Practices

**Document Processing:**
- Use `--content-type table` for spreadsheets, `--content-type diagram` for presentations with charts, `--content-type text` for text-heavy documents
- Use `--format markdown` for readable results, `--format json` for programmatic processing
- Adjust `--max-tokens` based on document complexity
- Command: `kg-extractor process <file_path> [options]`

**LangGraph Workflow:**
- Use `--similarity-threshold 0.3-0.5` for detailed analysis, `0.5-0.7` for balanced chunking, `0.7-1.0` for high-level overview
- Adjust `--min-chunk-size` and `--max-chunk-size` based on content (50-500 tokens for detailed, 500-1500 for overview)
- Use NVIDIA for document processing, OpenAI for chunking and triple extraction
- Review chunks for semantic coherence, triples for accuracy, causal links for plausibility
- Command: `kg-extractor langgraph <input_file> [options]`

**Triple Refinement:**
- Use Qdrant for entity resolution when dealing with large document sets
- Configure Qdrant with appropriate vector dimensions for your model
- Review refined triples for accuracy before graph building
- Command: `kg-extractor refine <input_file> [options]`

**Graph Building:**
- Ensure Neo4j instance is running and accessible
- Use appropriate constraints for entity types
- Monitor memory usage for large graphs
- Use `--with-schema` to validate against predefined schema
- Command: `kg-extractor graph <input_file>`

**Agent Usage:**
- Be specific with tasks, break complex tasks into smaller steps
- Use OpenAI models for best tool support
- Leverage triple extraction for knowledge graph construction
- Command: `kg-extractor --agent [options]`

**LangGraph Workflow:**
- Use for complex multi-step processes requiring state management
- Better for workflows with conditional logic and branching
- More overhead but provides better control and debugging
- Command: `kg-extractor langgraph <input_file> [options]`

**KG Reasoning:**
- Use natural language queries to explore the knowledge graph
- Adjust `--similarity-threshold` for entity matching (0.7-0.8 for strict, 0.6-0.7 for balanced)
- Use `--verbose` to see detailed metadata and query process
- Review query suggestions when no entity matches are found
- Command: `kg-reasoning query "<query>" [options]` or `kg-reasoning langgraph "<query>" [options]`

**Schema Validation:**
- Use `--with-schema` flag to enable schema validation
- Schema defined in `src/kg_extractor/utils/schema.md`
- Validates node types, relations, and enum values
- Rejects invalid entities and relationships during graph building
- Ensures knowledge graph conforms to expected structure
- Command: `kg-extractor langgraph <input_file> --with-schema`

## Schema Validation

The `--with-schema` flag enables schema validation to ensure extracted knowledge graphs conform to predefined structures.

### Schema File Format

The schema is defined in `src/kg_extractor/utils/schema.md` and includes:

**Node Types:**
- Tambon, Village, SocialCapital, Domain, Activity, TargetGroup, Impact, Evidence
- Resource, EnablingFactor, CommunityIssue, CapabilityDimension, CapabilityAssessment, Innovation, Actor

**Relations:**
- BELONGS_TO, LOCATED_IN, PERFORMS, TARGETS, PRODUCES, AFFECTS, SUPPORTS
- PARTICIPATES_IN, USES, ENABLED_BY, ADDRESSES, STRENGTHENS, CONNECTED_TO, EMERGES_FROM

**Enum Values:**
- SocialCapitalLevel: PERSON_FAMILY, SOCIAL_GROUP_COMMUNITY_ORG, AGENCY_RESOURCE_SOURCE, VILLAGE_COMMUNITY, TAMBON, NETWORK
- DomainCode: SOCIAL, ECONOMIC, ENVIRONMENT, HEALTH, GOVERNANCE
- ScopeLevel: VILLAGE, TAMBON, NETWORK
- ImpactType: DIRECT, INDIRECT, SHORT_TERM, LONG_TERM
- EvidenceStrength: STRONG, MODERATE, WEAK, CLAIMED
- ResourceType: PEOPLE, DATA, BUDGET, METHOD, TECHNOLOGY, FACILITY, KNOWLEDGE, NETWORK, POLICY

### Using Schema Validation

```bash
# Enable schema validation in LangGraph workflow
uv run kg-extractor langgraph document.pdf --with-schema

# Enable schema validation in graph building
uv run kg-extractor graph output/document_triples_refined.json --with-schema
```

### Customizing the Schema

To customize the schema for your domain:

1. **Edit the schema file:**
   ```bash
   # Edit src/kg_extractor/utils/schema.md
   # Add or modify node types, relations, and enum values
   ```

2. **Add new node types:**
   ```markdown
   ### YourNodeType
   | Field | Type | Required |
   |---|---|---|
   | `field_id` | string | yes |
   | `field_name` | string | yes |
   ```

3. **Define new relations:**
   ```markdown
   | Subject | Relation | Object |
   |---|---|---|
   | YourNodeType | YOUR_RELATION | AnotherNodeType |
   ```

4. **Add enum values:**
   ```markdown
   ### `YourEnum`
   ```
   VALUE1
   VALUE2
   VALUE3
   ```
   ```

5. **Test with schema validation:**
   ```bash
   uv run kg-extractor langgraph document.pdf --with-schema
   ```

### Schema Validation Behavior

When `--with-schema` is enabled:

- **Triple Extraction:** The LLM prompt includes schema constraints, guiding extraction to only use valid node types and relations
- **Node Validation:** Invalid node types are rejected during graph building with warning messages
- **Relation Validation:** Invalid relations are rejected during graph building with warning messages
- **Enum Validation:** Enum field values are validated against schema definitions
- **Strict Compliance:** Only schema-compliant entities and relationships are stored in Neo4j

### Schema Validation Benefits

- **Data Quality:** Ensures knowledge graph conforms to expected structure
- **Consistency:** Maintains consistent entity types and relationships across documents
- **Queryability:** Enables predictable queries with known node types and relations
- **Domain Specificity:** Allows customization for specific domains and use cases
- **Error Prevention:** Prevents invalid data from entering the knowledge graph

## Future Enhancements

- Additional file formats (EPUB, HTML, etc.)
- Advanced chunking strategies (hybrid LLM and embeddings)
- Knowledge graph visualization tools
- Entity resolution and cross-document linking
- Graph database integration (Neo4j, ArangoDB) - **Partially implemented**
- Batch processing for multiple files
- Custom prompts and output templates
- Performance optimization with caching
- REST API and web interface
- Advanced Qdrant features (hybrid search, filtering)
- Neo4j query interface and visualization
- Real-time graph updates and streaming

## Troubleshooting

**API Key Issues:** Ensure `.env` file exists in project root with appropriate API key (OPENAI_API_KEY, NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY).

**Command Not Found:** Make sure you're using the correct command structure:
- Document processing: `kg-extractor process <file_path>` or just `kg-extractor <file_path>` (backward compatible)
- LangGraph: `kg-extractor langgraph <input_file>`
- Refine: `kg-extractor refine <input_file>`
- Graph: `kg-extractor graph <input_file>`
- Agent: `kg-extractor --agent`
- KG Reasoning: `kg-reasoning query "<query>"` or `kg-reasoning langgraph "<query>"`

**Agent Mode Issues:** Use OpenAI models (gpt-4o-mini) for best tool support. Free models may have rate limits.

**File Processing Errors:** Check file path, format support, and read permissions. Password-protected files not supported.

**API Errors:** Verify API key, check rate limits, ensure internet connection, verify model name.

**Content Quality Issues:** Increase `--max-tokens` for incomplete extraction, ensure source documents are clear and readable.

**LangGraph Workflow Issues:** Adjust `--similarity-threshold` for chunking (lower = more chunks). Ensure chunks contain meaningful content for triple extraction. Causal links only extracted when explicitly mentioned.

**Qdrant/Neo4j Issues:** Ensure QDRANT_URL, QDRANT_API_KEY, NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are set in environment variables when using refine or graph commands.

**Schema Validation Issues:** Ensure `src/kg_extractor/utils/schema.md` exists and is properly formatted. Invalid schema definitions will cause validation errors. Review schema.md for correct node types, relations, and enum values. Use `--with-schema` only when you want strict schema compliance.

**Schema Customization:** To customize the schema for your domain:
1. Edit `src/kg_extractor/utils/schema.md`
2. Add or modify node types with their required fields
3. Define valid relations between node types
4. Specify enum values for constrained fields
5. Test with `--with-schema` flag to validate extraction

**KG Reasoning Issues:** Ensure both Qdrant and Neo4j are configured and accessible. Entity matching requires Qdrant to have indexed entities from previous triple extraction. Query results depend on graph structure and data quality. Adjust `--similarity-threshold` for entity matching (lower = more matches, higher = stricter matching).

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