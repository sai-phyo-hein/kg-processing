# Knowledge Graph Processing

Two-module system for extracting knowledge from documents and reasoning over the resulting knowledge graphs.

## Modules

### kg-extractor
Extract structured content from documents → semantic chunks → knowledge graph triples.

**Features:**
- Multi-format support: PDF, DOCX, PPTX, XLSX, PNG, JPEG, WEBP, GIF, BMP
- Multi-API providers: OpenAI, NVIDIA, OpenRouter, Groq, Google
- LLM-based semantic chunking with topic boundary detection
- OpenIE triple extraction with causal discovery and temporal metadata
- Triple refinement with Qdrant (entity resolution, canonical ID assignment)
- Neo4j graph building with schema validation (optional)

**Quick Start:**
```bash
# Process document and build knowledge graph
uv run kg-extractor document.pdf

```

**Key Options:**
- `--chunk-granularity`: Semantic chunking granularity (0.0-1.0, default: 0.5)
- `--similarity-threshold`: Entity resolution threshold (0.0-1.0, default: 0.95)
- `--until <step>`: Stop after specific step (document_parsing, metadata_extraction, semantic_chunking, triple_extraction, triple_refining, graph_building)
- `--pages`: Process specific pages (e.g., '1,3-5,7')

**Debug Modes:**
```bash
# Refine existing triples
uv run kg-extractor document.pdf --refine-triples

# Build graph from existing refined triples
uv run kg-extractor document.pdf --build-graph
```

### kg-reasoning
Query knowledge graphs using multi-agent reasoning (Orchestrator → Workers → Synthesizer).

**Features:**
- Multi-agent workflow: Orchestrator plans strategies, Workers execute searches, Synthesizer answers
- Entity extraction and matching against Qdrant vector database
- Neo4j Cypher query execution
- Natural language answer synthesis

**Quick Start:**
```bash
# Query the knowledge graph
uv run kg-reasoning query "What drives Thailand ecommerce?"

# With verbose output
uv run kg-reasoning query "How does debt affect company performance?" --verbose

# Save results to file
uv run kg-reasoning query "What are market volatility causes?" --output result.json
```

**Key Options:**
- `--llm-provider`: LLM provider (default: openai)
- `--orchestrator-model`: Orchestrator agent model (default: gpt-4o)
- `--worker-model`: Worker agents model (default: gpt-4o-mini)
- `--synthesizer-model`: Synthesizer agent model (default: gpt-4o)
- `--output`: Save results to JSON file
- `--verbose`: Show workflow metadata (strategies, entities found, execution time)

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

# PARSING_PROVIDER=
# CHUNKING_PROVIDER=
# TRIPLET_PROVIDER=
# REFINEMENT_PROVIDER=
# EMBEDDING_PROVIDER=

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# PARSING_MODEL=
# CHUNKING_MODEL=
# TRIPLET_MODEL=
# REFINEMENT_MODEL=
# OPENAI_EMBEDDING_MODEL=

# Reasoning agents
# REASONING_PROVIDER=
# ORCHESTRATOR_MODEL=
# SYNTHESIZER_MODEL=
```

## Usage

### Quick Start

```bash
# Run complete LangGraph workflow (recommended)
uv run kg-extractor langgraph document.pdf
```

**Supported Providers:**
- `nvidia` - NVIDIA API (default, good for vision tasks)
- `openrouter` - OpenRouter API (free options available)
- `openai` - OpenAI API (high quality, gpt-4o recommended)
- `groq` - Groq API (fast inference)
- `google` - Google Gemini API (gemini-3.1-flash-lite-preview)

**Recommended Models:**
- NVIDIA: `microsoft/phi-4-multimodal-instruct` (vision), `google/gemma-3-27b-it` (general), `meta/llama-4-maverick-17b-128e-instruct`
- OpenRouter: `google/gemma-4-31b-it:free` (free), `openai/gpt-oss-120b:free` (free)
- OpenAI: `gpt-4o` (best quality), `gpt-4o-mini` (cost-effective)
- Groq: `openai/gpt-oss-120b` (very fast)
- Google: `google/gemini-3.1-flash-lite-preview` (fast, cost-effective)

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
