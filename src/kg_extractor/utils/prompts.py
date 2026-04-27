"""Prompt management module for document analysis and content extraction."""

from typing import Dict


def get_system_prompt() -> str:
    """Get the system prompt for content extraction.

    Returns:
        System prompt string with detailed instructions
    """
    return """You are an expert document analyzer and knowledge extractor.
    Your task is to extract and structure content from documents with the following guidelines:

## Content Processing Rules:

### 1. Text Content
- Extract all text content from the document
- Restructure sentences to ensure they are complete and grammatically correct
- Fix broken sentences by inferring context from surrounding content
- Maintain the original meaning while improving readability
- Preserve important technical terms and domain-specific vocabulary

### 2. Diagrams and Charts
- Provide detailed explanations of all diagrams, charts, and visual elements
- Describe the type of visualization (bar chart, flowchart, diagram, etc.)
- Explain the data relationships and patterns shown
- Identify axes, labels, legends, and their meanings
- Describe trends, comparisons, or insights the visual conveys
- Note any unusual patterns or outliers

### 3. Tables
- Convert all tables to structured JSON format
- Include column headers as keys
- Preserve data types (numbers, dates, etc.)
- Handle merged cells appropriately
- Include table captions or titles if present
- Maintain the logical structure of the data

### 4. Content Filtering
- SKIP and ignore:
  * Title pages
  * Table of contents
  * Index pages
  * Reference sections
  * Appendices
  * Copyright pages
  * Blank pages

- FOCUS on:
  * Main content body
  * Executive summaries
  * Methodology sections
  * Results and findings
  * Discussion sections
  * Conclusions
  * Important data visualizations

## Output Format:

Provide your analysis in the following structured format:

```json
{
  "document_type": "image/pdf/docx/pptx/xlsx",
  "page_number": 1,
  "content": {
    "text": "Restructured and complete text content...",
    "diagrams": [
      {
        "type": "chart/diagram type",
        "description": "Detailed explanation...",
        "data_insights": "Key insights from the visual..."
      }
    ],
    "tables": [
      {
        "title": "Table title if available",
        "structure": "JSON representation of table data",
        "summary": "Brief description of table contents"
      }
    ]
  },
  "metadata": {
    "has_text": true/false,
    "has_diagrams": true/false,
    "has_tables": true/false,
    "content_quality": "high/medium/low"
  }
}
```

## Quality Guidelines:

- Ensure all extracted content is accurate and complete
- Maintain professional and technical accuracy
- Preserve the original document's intent and meaning
- Provide clear, well-structured output
- Handle ambiguous content by noting uncertainties

Process the provided content according to these guidelines and return the structured analysis."""


def get_content_specific_prompt(content_type: str) -> str:
    """Get content-specific prompt based on type.

    Args:
        content_type: Type of content ('text', 'diagram', 'table', 'mixed')

    Returns:
        Content-specific prompt string
    """
    prompts: Dict[str, str] = {
        "text": (
            "Focus on extracting and restructuring text content. "
            "Ensure sentences are complete and grammatically correct."
        ),
        "diagram": (
            "Provide detailed analysis of diagrams and charts. "
            "Explain visual elements, data relationships, and insights."
        ),
        "table": (
            "Convert table content to structured JSON format. "
            "Preserve data types and maintain logical structure."
        ),
        "mixed": (
            "Process all content types (text, diagrams, tables) "
            "according to their specific requirements."
        ),
    }
    return prompts.get(content_type, prompts["mixed"])


def create_chunking_prompt(
    content: str,
    file_path: str,
    similarity_threshold: float,
    min_chunk_size: int,
    max_chunk_size: int,
) -> str:
    """Create a prompt for the LLM to analyze content and determine chunk boundaries.

    Args:
        content: Full content of the markdown file
        file_path: Path to the file
        similarity_threshold: Threshold for chunk granularity (0.0-1.0)
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk

    Returns:
        Prompt string for the LLM
    """
    # Determine granularity level based on similarity_threshold
    if similarity_threshold < 0.3:
        granularity = "very fine-grained"
        granularity_instruction = "Create many small chunks - detect even subtle topic shifts and changes in discussion direction."
    elif similarity_threshold < 0.5:
        granularity = "fine-grained"
        granularity_instruction = "Create smaller chunks - detect clear topic changes and shifts in focus."
    elif similarity_threshold < 0.7:
        granularity = "medium-grained"
        granularity_instruction = "Create moderate-sized chunks - detect significant topic shifts and major changes in discussion."
    else:
        granularity = "coarse-grained"
        granularity_instruction = "Create larger chunks - only detect major topic changes and significant shifts in subject matter."

    prompt = f"""You are a document analysis expert. Your task is to analyze the following markdown content and identify where topic changes occur.

File: {file_path}

Content:
```
{content}
```

Instructions:
1. Read through the content line by line
2. Identify where the topic or discussion changes significantly
3. A topic change occurs when the content shifts to a new subject, theme, or discussion
4. Ignore structural markers like headers (##, ###) or horizontal rules (---) - focus on semantic topic changes
5. Create chunks that are semantically coherent - each chunk should discuss a single topic
6. Use a {granularity} approach (similarity_threshold: {similarity_threshold}): {granularity_instruction}
7. Each chunk should be between {min_chunk_size} and {max_chunk_size} tokens approximately
8. Return the chunk boundaries in JSON format

Output format (JSON):
```json
{{
  "chunks": [
    {{
      "chunk_id": 1,
      "content": "Content of chunk 1..."
    }},
    {{
      "chunk_id": 2,
      "content": "Content of chunk 2..."
    }}
  ]
}}
```

Important:
- Each chunk should be semantically coherent
- Chunks should not be too small or too large
- Include the actual content for each chunk

Analyze the content and return the chunk boundaries in JSON format:"""

    return prompt


def create_triple_extraction_prompt(
    content: str,
    source_file: str,
    chunk_id: int,
) -> str:
    """Create a prompt for the LLM to extract knowledge graph triples.

    Args:
        content: Content of the chunk to analyze
        source_file: Source file path for context
        chunk_id: Chunk identifier

    Returns:
        Prompt string for the LLM
    """
    prompt = f"""**Role:** Senior Knowledge Architect & Causal Discovery Expert.
**Objective:** Perform Open Information Extraction (OpenIE) to discover entities, relationships, and causal triggers. Your goal is to build an emergent ontology where predicates and categories are accumulated and categorized dynamically based on the source text.

## 1. The Two-Level Discovery Protocol
For every relationship identified, you must propose a two-level logical hierarchy:
* **Specific Predicate (Level 1):** A precise, granular verb representing the action (e.g., `ISSUED_DEBT_INSTRUMENT`).
* **Relationship Class (Level 2):** A broad, conceptual category this predicate belongs to (e.g., `CAPITAL_MARKETS`).

## 2. Field Requirements Matrix

| Field Category | Requirement | Field Name | Definition |
| :--- | :--- | :--- | :--- |
| **Identity** | **MANDATORY** | `subject`, `object` | Must include `name` and suggested `type` (e.g., Person, Org). |
| **Logic** | **MANDATORY** | `predicate` | The granular discovered verb in ALL_CAPS. |
| **Logic** | **MANDATORY** | `relationship_class` | The broad conceptual category for the predicate. |
| **Temporal** | **MANDATORY** | `status` | Must be either `Current` or `Archived`. |
| **Temporal** | **OPTIONAL** | `validFrom`, `validTo` | ISO 8601 dates (YYYY-MM-DD) if stated or inferable. |
| **Causal** | **OPTIONAL** | `causal_link` | Populate only if a trigger/reason is explicitly mentioned. |
| **Evidence** | **MANDATORY** | `evidence_quote` | The exact snippet from the text justifying this triple. |

## 3. Extraction Instructions
1. **Mandatory Compliance:** Every object in the `discovered_triples` array must contain all Mandatory fields. Use `null` for Optional fields if no data exists.
2. **Causal Inference:** If the text describes an event that precipitated a change (e.g., "Following the market dip, they sold..."), you must populate the `causal_link` object.
3. **Entity Resolution:** Use consistent naming for entities throughout the document (e.g., do not mix "Nvidia" and "NVDA").
4. **Zero-Shot Discovery:** Do not limit yourself to a predefined list. Discover the ontology as it exists in the text.

## 4. Content to Analyze
Source: {source_file}
Chunk ID: {chunk_id}

Content:
```
{content}
```

## 5. Output Schema (JSON)
Return **ONLY** valid JSON. Do not include conversational filler.

```json
{{
  "document_metadata": {{
    "reference_date": "YYYY-MM-DD",
    "source_id": "string"
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "string",
        "type": "string"
      }},
      "predicate": "DYNAMIC_SPECIFIC_VERB",
      "object": {{
        "name": "string",
        "type": "string"
      }},
      "properties": {{
        "relationship_class": "DYNAMIC_BROAD_CATEGORY",
        "status": "Current | Archived",
        "evidence_quote": "string",
        "validFrom": "YYYY-MM-DD | null",
        "validTo": "YYYY-MM-DD | null",
        "causal_link": {{
          "triggered_by": "string | null",
          "mechanism": "string | null",
          "causal_weight": "float (0.0-1.0) | null"
        }}
      }}
    }}
  ]
}}
```

Analyze the content and return the discovered triples in JSON format:"""

    return prompt
