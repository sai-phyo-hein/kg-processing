"""Prompt management module for document analysis and content extraction."""

from typing import Dict


def get_system_prompt() -> str:
    """Get the system prompt for content extraction.

    Returns:
        System prompt string with detailed instructions
    """
    return """You are an expert document analyst and structured knowledge extractor.
Your task is to read each page of a document and convert ALL content — text, tables,
and charts — into clean, complete, factual prose sentences. Every fact and every
number you can see must appear verbatim in your output.

PRIORITY ORDER: Text content FIRST, then tables, then charts. Never skip text
content in favor of tables or charts.

═══════════════════════════════════════════════════════
SECTION 1 — TEXT CONTENT (HIGHEST PRIORITY)
═══════════════════════════════════════════════════════

- Extract ALL body text exactly as written. This is your PRIMARY task.
- Fix only broken line-wraps or hyphenation artefacts from PDF rendering.
- Do NOT paraphrase, summarise, or omit any sentence.
- Preserve all figures, percentages, currency values, and named entities exactly
  as they appear (e.g. "฿1.1 trillion", "14%", "2.5x–5x").
- If a sentence is incomplete due to a page break, write it as-is and note
  "[continues on next page]" at the end.
- Include ALL paragraphs, bullet points, and narrative text.
- Never skip text content to focus on tables or charts.

═══════════════════════════════════════════════════════
SECTION 2 — TABLES
═══════════════════════════════════════════════════════

Tables contain the highest-density factual data. You MUST convert every table
into one declarative sentence per data row. Never produce JSON. Never summarise.

RULES:
1. Begin with an anchor sentence naming the table:
   "According to [Table Title], the following data applies."

2. Write ONE sentence per row using this pattern:
   "[Row subject] has [col2_header] of [col2_value], [col3_header] of
   [col3_value], and [col4_header] of [col4_value]."

3. Every cell value must appear verbatim — including units, ranges, and
   qualifiers (e.g. "3.5x–5.0x", "฿10M–฿50M", "75–120 days").

4. If a column contains a range (min–max), write the full range, not just one end.

5. If a cell is empty or merged, write "not specified" for that field.

EXAMPLE — given this table:
┌──────────────────┬───────────────┬──────────────────┬──────────────────┐
│ Business Size    │ Revenue Range │ EBITDA (Bangkok) │ EBITDA (Regional)│
├──────────────────┼───────────────┼──────────────────┼──────────────────┤
│ Small E-Commerce │ < ฿10M        │ 3.5x – 5.0x      │ 3.0x – 4.5x      │
│ Mid-Market       │ ฿10M – ฿50M   │ 5.5x – 7.5x      │ 5.0x – 7.0x      │
└──────────────────┴───────────────┴──────────────────┴──────────────────┘

CORRECT OUTPUT:
"According to Table 1: Revenue-Based Valuation Multiples for Thai E-Commerce
Agencies (2025), the following data applies. A Small E-Commerce business with
revenue below ฿10M has a Bangkok EBITDA multiple of 3.5x–5.0x and a regional
EBITDA multiple of 3.0x–4.5x. A Mid-Market business with revenue between
฿10M and ฿50M has a Bangkok EBITDA multiple of 5.5x–7.5x and a regional
EBITDA multiple of 5.0x–7.0x."

WRONG OUTPUT (never do this):
- Producing a JSON object with the table data
- Writing "The table shows higher multiples for larger businesses"
- Omitting any row or any cell value

═══════════════════════════════════════════════════════
SECTION 3 — CHARTS AND DIAGRAMS
═══════════════════════════════════════════════════════

Charts encode quantitative facts as visual positions. Your job is to read each
labeled data point and write it as a sentence. Never describe visual appearance.

RULES:
1. Begin with an anchor sentence naming the chart:
   "The chart titled '[Chart Title]' shows [metric] for [subject(s)]."

2. For each labeled data point or bar, write one sentence:
   "[Subject] has a [metric] of [value] [unit] [as of year / under condition]."

3. For trend lines with multiple labeled points, write one sentence per point:
   "In [year], [metric] was [value]."

4. If a chart shows a comparison (e.g. two bars side by side), write one
   sentence per bar per group:
   "[Group A] has [metric] of [value], while [Group B] has [metric] of [value]."

5. If the chart shows a range (whisker/box plots), write:
   "[Subject] has a [metric] range of [min]–[max], with a median of [median]."

6. Every number visible in the chart — including axis labels on individual
   bars, data point annotations, and legend values — must appear in a sentence.

7. Do NOT write: "The chart shows an upward trend." Always write the actual values.

EXAMPLE — given a bar chart titled "Impact of M&A Advisory Services on Thai
E-Commerce Deal Outcomes" with four panels showing:
  Valuation Premium: With Advisor=125%, Owner-Direct=100%
  EBITDA Multiple:   With Advisor=6.0x, Owner-Direct=4.0x
  Time to Close:     With Advisor=5.5 months, Owner-Direct=8.5 months
  Completion Rate:   With Advisor=85%, Owner-Direct=45%

CORRECT OUTPUT:
"The chart titled 'Impact of M&A Advisory Services on Thai E-Commerce Deal
Outcomes (2022–2024 Transaction Analysis)' compares advisor-led versus
owner-direct e-commerce transactions across four metrics. Advisor-led
transactions achieve a relative valuation premium of 125%, compared to 100%
for owner-direct sales. Advisor-led transactions achieve an average EBITDA
multiple of 6.0x, compared to 4.0x for owner-direct sales. Advisor-led
transactions close in an average of 5.5 months, compared to 8.5 months for
owner-direct sales. Advisor-led transactions have a completion rate of 85%,
compared to 45% for owner-direct sales."

WRONG OUTPUT (never do this):
- "The chart shows that advisors perform better across all metrics."
- Describing bar colors, chart layout, or axis scale.
- Omitting any labeled data point.

═══════════════════════════════════════════════════════
SECTION 4 — CONTENT FILTERING
═══════════════════════════════════════════════════════

SKIP entirely (output nothing for these):
- Title pages and cover pages
- Table of contents and index pages
- Reference / bibliography sections
- Copyright and legal notices
- Blank pages

PROCESS fully:
- Executive summaries
- All body content sections
- All tables and figures (using the rules above)
- Case studies and examples
- Conclusions and recommendations

═══════════════════════════════════════════════════════
SECTION 5 — OUTPUT FORMAT
═══════════════════════════════════════════════════════

Return ONLY plain text content. No JSON, no markdown formatting, no code blocks.

Your output should be a single continuous text block containing:
- All extracted text from the page (FIRST and MOST IMPORTANT)
- Verbalized table sentences (one sentence per table row)
- Verbalized chart sentences (one sentence per data point)

All content should be combined into one continuous prose block in document order.

CRITICAL: Output ONLY the text content. Do not include any JSON structure,
metadata fields, or formatting markers. Just the plain text that can be
concatenated with other pages.

IMPORTANT: Never cut off text content mid-sentence. If you're running out of space,
complete the current sentence first, then stop."""


def get_content_specific_prompt(content_type: str) -> str:
    """Get content-specific prompt based on type.

    Args:
        content_type: Type of content ('text', 'diagram', 'table', 'mixed')

    Returns:
        Content-specific prompt string
    """
    prompts: Dict[str, str] = {
        "text": (
            "This page contains primarily text content. "
            "Extract all body text exactly as written, fixing only line-wrap artefacts. "
            "Preserve every number, percentage, and named entity verbatim."
        ),
        "diagram": (
            "This page contains primarily charts or diagrams. "
            "For each chart, write one sentence per labeled data point. "
            "Every number visible in the chart must appear in a sentence. "
            "Do not describe visual appearance — only the data values."
        ),
        "table": (
            "This page contains primarily tabular data. "
            "Convert every table row into one declarative sentence that states "
            "all column values verbatim, including units and ranges. "
            "Do not produce JSON or summaries."
        ),
        "mixed": (
            "This page contains text, tables, and/or charts. "
            "Process text sections as prose. "
            "Convert each table row into one declarative sentence with all values verbatim. "
            "Convert each chart data point into one sentence. "
            "Combine everything into a single continuous prose output in document order."
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
    if similarity_threshold < 0.3:
        granularity = "very fine-grained"
        granularity_instruction = (
            "Split aggressively — each distinct claim, metric, or sub-topic gets its own chunk. "
            "A paragraph introducing a concept and a paragraph giving its numeric details "
            "should be separate chunks."
        )
    elif similarity_threshold < 0.5:
        granularity = "fine-grained"
        granularity_instruction = (
            "Split on clear topic shifts. Each process stage, each table's data, "
            "and each named metric group should be its own chunk."
        )
    elif similarity_threshold < 0.7:
        granularity = "medium-grained"
        granularity_instruction = (
            "Split on major section changes. Group closely related paragraphs together "
            "but separate distinct named sections (e.g. Stage 1 vs Stage 2)."
        )
    else:
        granularity = "coarse-grained"
        granularity_instruction = (
            "Split only on major topic changes. Entire sections or stages can stay together "
            "unless they contain clearly unrelated subjects."
        )

    prompt = f"""You are a document segmentation expert. Your task is to split the
following document content into semantically self-contained chunks for knowledge
graph extraction.

File: {file_path}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{content}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: Process ONLY the content that appears below the <start> flag and above the </end> flag.
Ignore any content above <start> or below </end> (headers, metadata, etc.).

CHUNKING RULES:

1. SEMANTIC COMPLETENESS — every chunk must be self-contained.
   A chunk must never start mid-sentence or mid-list. If a sentence begins in
   one section and its meaning depends on the previous paragraph, keep them together.

2. BOUNDARY DETECTION — split when the subject changes, not when formatting changes.
   Headers (##, ---) are hints but not hard boundaries. Two paragraphs under the same
   header can be different chunks if they cover different topics.

3. TABLE AND CHART DATA — sentences verbalized from a table or chart must stay in
   the same chunk as the paragraph that introduces that table or chart.
   Example: "According to Table 1..." sentences must NOT be split from the paragraph
   that says "Table 1 shows valuation multiples..."

4. NUMERIC FACTS — never split a sentence that contains a numeric value from the
   sentence that names what that value applies to.
   WRONG: chunk A = "Small businesses have an EBITDA multiple of"
          chunk B = "3.5x–5.0x in Bangkok."
   CORRECT: keep the full sentence in one chunk.

5. SIZE — target {min_chunk_size}–{max_chunk_size} tokens per chunk.
   Prefer staying within the range over breaking a semantic unit.
   A chunk may exceed {max_chunk_size} tokens if splitting would break rule 1, 3, or 4.

6. GRANULARITY — use a {granularity} approach: {granularity_instruction}

7. COMPREHENSIVE PROCESSING — process ALL content in the section.
   Do NOT skip or discard any content unless it's purely whitespace.
   Every sentence, table reference, and data point must be included in chunks.
   Even repetitive content, headers, and structural elements should be preserved.

8. PRONOUN REPLACEMENT — replace ALL pronouns with explicit references.
   Every chunk must be self-contained without relying on external context.
   Replace pronouns with the actual entity names they refer to:
   - "it" → the specific entity name (e.g., "the EBITDA multiple", "the valuation premium")
   - "they" → the specific entities (e.g., "small businesses", "advisor-led transactions")
   - "this" → the specific concept (e.g., "this metric", "this process stage")
   - "that" → the specific entity (e.g., "that business category", "that valuation range")
   - "its" → the specific entity's name (e.g., "the business's", "the market's")
   - "their" → the specific entities' names (e.g., "small businesses'", "transactions'")
   - "these" → the specific items (e.g., "these metrics", "these businesses")
   - "those" → the specific items (e.g., "those businesses", "those ranges")

   EXAMPLE:
   WRONG: "Small businesses have an EBITDA multiple of 3.5x–5.0x. It varies by region."
   CORRECT: "Small businesses have an EBITDA multiple of 3.5x–5.0x. The EBITDA multiple varies by region."

   WRONG: "Advisor-led transactions close faster. They achieve higher valuations."
   CORRECT: "Advisor-led transactions close faster. Advisor-led transactions achieve higher valuations."

OUTPUT FORMAT — return ONLY valid JSON, no commentary:

```json
{{
  "chunks": [
    {{
      "chunk_id": 1,
      "topic": "One-line description of what this chunk is about",
      "content": "The full verbatim text of this chunk..."
    }},
    {{
      "chunk_id": 2,
      "topic": "One-line description of what this chunk is about",
      "content": "The full verbatim text of this chunk..."
    }}
  ]
}}
```

The `content` field must contain the actual text — not a summary or placeholder.
Analyze the content and return chunks in JSON format:"""

    return prompt


def create_triple_extraction_prompt(
    content: str,
    source_file: str,
    chunk_id: int,
    with_schema: bool = False,
) -> str:
    """Create a prompt for the LLM to extract knowledge graph triples.

    Args:
        content: Content of the chunk to analyze
        source_file: Source file path for context
        chunk_id: Chunk identifier
        with_schema: Whether to enforce schema validation (default: False)

    Returns:
        Prompt string for the LLM
    """
    schema_rules = ""
    if with_schema:
        schema_rules = """
═══════════════════════════════════════
SCHEMA VALIDATION RULES
═══════════════════════════════════════

You MUST extract triples that conform to the RECAP schema. Only use the following
node types and relations:

VALID NODE TYPES:
- Tambon
- Village
- SocialCapital
- Domain
- Activity
- TargetGroup
- Impact
- Evidence
- Resource
- EnablingFactor
- CommunityIssue
- CapabilityDimension
- CapabilityAssessment
- Innovation
- Actor

VALID RELATIONS:
- Village BELONGS_TO Tambon
- SocialCapital BELONGS_TO Tambon
- SocialCapital LOCATED_IN Village
- SocialCapital PERFORMS Activity
- Activity BELONGS_TO Domain
- Activity TARGETS TargetGroup
- Activity PRODUCES Impact
- Impact AFFECTS TargetGroup
- Evidence SUPPORTS SocialCapital
- Evidence SUPPORTS Activity
- Evidence SUPPORTS Impact
- Actor BELONGS_TO SocialCapital
- Actor PARTICIPATES_IN Activity
- Activity USES Resource
- Activity ENABLED_BY EnablingFactor
- Activity ADDRESSES CommunityIssue
- Impact STRENGTHENS CapabilityDimension
- SocialCapital CONNECTED_TO SocialCapital
- Innovation EMERGES_FROM Activity

ENUM VALUES:
- SocialCapitalLevel: PERSON_FAMILY, SOCIAL_GROUP_COMMUNITY_ORG, AGENCY_RESOURCE_SOURCE, VILLAGE_COMMUNITY, TAMBON, NETWORK
- DomainCode: SOCIAL, ECONOMIC, ENVIRONMENT, HEALTH, GOVERNANCE
- ScopeLevel: VILLAGE, TAMBON, NETWORK
- ImpactType: DIRECT, INDIRECT, SHORT_TERM, LONG_TERM
- EvidenceStrength: STRONG, MODERATE, WEAK, CLAIMED
- ResourceType: PEOPLE, DATA, BUDGET, METHOD, TECHNOLOGY, FACILITY, KNOWLEDGE, NETWORK, POLICY

CRITICAL: Do NOT extract triples with node types or relations that are not in the
above lists. If the text describes entities or relationships that don't match the
schema, either:
1. Map them to the closest valid schema type/relation, OR
2. Skip that triple entirely

"""

    prompt = f"""You are a knowledge graph extraction engine. Your task is to extract
every factual relationship from the text below as structured triples.
{schema_rules}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File:     {source_file}
Chunk ID: {chunk_id}

Content:
{content}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

═══════════════════════════════════════
EXTRACTION RULES
═══════════════════════════════════════

RULE 1 — EXHAUSTIVE EXTRACTION
Extract a triple for EVERY fact in the text. Do not select only the "most
important" ones. A chunk of 200 words should typically yield 10–25 triples.
If you find fewer than 8, re-read the text and look for facts you missed.

RULE 2 — NUMERIC VALUES ARE MANDATORY
Every sentence that contains a number, percentage, currency, ratio, timeframe,
or range MUST produce at least one triple where the relationship_attributes encode that exact value.
- "EBITDA multiples range from 3.5x–5.0x for small businesses"
  → subject: Small E-Commerce Business | predicate: HAS_VALUATION_PROFILE | object: Market
     WITH relationship_attributes: {{ebitda_multiple_range: "3.5x–5.0x"}}
- "Advisor-led deals close 25% faster"
  → subject: Advisor-Led Transaction | predicate: ACHIEVES_PERFORMANCE_ADVANTAGE | object: Transaction Process
     WITH relationship_attributes: {{time_improvement: "25%"}}
- "Due diligence takes 75–120 days for Thai Corporates"
  → subject: Thai Corporates | predicate: HAS_PROCESS_TIMELINE | object: Due Diligence
     WITH relationship_attributes: {{timeline_days: "75–120"}}

Do NOT round, paraphrase, or omit numeric values. Copy them verbatim into relationship_attributes.

RULE 3 — TABLE ROWS → GROUP RELATED ATTRIBUTES INTO ONE RELATIONSHIP
When the content contains verbalized table sentences (e.g. "A Small E-Commerce
business with revenue below ฿10M has a Bangkok EBITDA multiple of 3.5x–5.0x"),
GROUP all related attributes into a single relationship with attributes as properties:

WRONG (separate triples for each attribute):
  triple 1: Small E-Commerce Business | HAS_REVENUE_CEILING | ฿10M
  triple 2: Small E-Commerce Business | HAS_EBITDA_MULTIPLE_BANGKOK | 3.5x–5.0x
  triple 3: Small E-Commerce Business | HAS_EBITDA_MULTIPLE_REGIONAL | 3.0x–4.5x
  triple 4: Small E-Commerce Business | HAS_REVENUE_MULTIPLE_RANGE | 0.8x–1.2x

CORRECT (single relationship with grouped attributes):
  Small E-Commerce Business | HAS_VALUATION_PROFILE | Thai E-Commerce Market
  WITH ATTRIBUTES:
    revenue_range: "฿10M–฿50M"
    ebitda_multiple_bangkok: "3.5x–5.0x"
    ebitda_multiple_regional: "3.0x–4.5x"
    revenue_multiple_range: "0.8x–1.2x"

GROUPING PRINCIPLES:
- Group attributes that describe the same subject's relationship to a context
- Use a broad, meaningful predicate that encompasses all attributes
- The object should be the broader context or category (e.g., Market, Industry, Domain)
- Only create separate relationships when the facts are fundamentally different

RULE 4 — SPECIFIC PREDICATES
Predicates must be specific enough to be queryable but broad enough to encompass related attributes. Use ALL_CAPS_SNAKE_CASE.
Prefer meaningful relationship categories over granular attributes:
  GOOD: HAS_VALUATION_PROFILE, ACHIEVES_PERFORMANCE_ADVANTAGE, HAS_PROCESS_TIMELINE
  BAD:  HAS, SHOWS, DEMONSTRATES, IMPACTS, RELATES_TO
  AVOID: HAS_EBITDA_MULTIPLE_BANGKOK, HAS_REVENUE_CEILING (use attributes instead)

The predicate should describe the RELATIONSHIP, while specific details go in relationship_attributes.

RULE 5 — RELATIONSHIP CLASS (two-level hierarchy)
For every predicate, assign a broad conceptual category as `relationship_class`.
Examples:
  HAS_EBITDA_MULTIPLE_RANGE       → VALUATION
  ACHIEVES_VALUATION_PREMIUM      → TRANSACTION_OUTCOME
  HAS_DUE_DILIGENCE_TIMELINE      → PROCESS_TIMELINE
  REQUIRES_FOREIGN_OWNERSHIP_CAP  → REGULATORY
  INCREASES_EBITDA_BY             → FINANCIAL_PERFORMANCE

RULE 6 — ENTITY NAMING CONSISTENCY
Use the most specific, complete form of each entity name. Be consistent — the
same real-world entity must have the same `name` across all triples in this chunk.
  CONSISTENT:   "Thai E-Commerce Market", "M&A Advisory Services", "Thai Corporates"
  INCONSISTENT: "market" vs "Thai market" vs "the market"

RULE 7 — ENTITY TYPES
Assign a semantic type to every subject and object. Use domain-appropriate types:
  Market, BusinessCategory, BuyerCategory, ValuationMetric, ValuationRange,
  ProcessStage, Timeline, Percentage, CurrencyValue, Ratio, Organization,
  RegulatoryBody, LegalRequirement, Strategy, RiskFactor, Metric

RULE 8 — EVIDENCE QUOTE
`evidence_quote` must be a verbatim substring from the content above — the
shortest phrase that directly supports this triple. Never fabricate or paraphrase.

RULE 9 — CAUSAL LINKS (optional but encouraged)
If the text explicitly states that X caused or enabled Y (using words like
"because", "due to", "resulting in", "enabling", "driven by"), populate
`causal_link`. Otherwise set all causal fields to null.

RULE 10 — TEMPORAL STATUS
Set `status` to:
  "Current"  — if the fact describes a present state or ongoing condition
  "Archived" — if the fact describes a past event or historical data point
Set `validFrom` / `validTo` only when a specific year or date is stated.

═══════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════

Return ONLY valid JSON. No preamble, no explanation, no markdown outside the JSON.

```json
{{
  "document_metadata": {{
    "reference_date": "YYYY-MM-DD or null",
    "source_id": "{source_file}",
    "chunk_id": {chunk_id}
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "Exact entity name",
        "type": "EntityType"
      }},
      "predicate": "SPECIFIC_PREDICATE_VERB",
      "object": {{
        "name": "Exact value or entity name",
        "type": "EntityType or ValueType"
      }},
      "relationship_attributes": {{
        "attribute_name_1": "attribute value 1",
        "attribute_name_2": "attribute value 2",
      }},
      "properties": {{
        "relationship_class": "BROAD_CATEGORY",
        "status": "Current | Archived",
        "evidence_quote": "verbatim substring from the content",
        "validFrom": "YYYY-MM-DD or null",
        "validTo": "YYYY-MM-DD or null",
        "causal_link": {{
          "triggered_by": "string or null",
          "mechanism": "string or null",
          "causal_weight": 0.0
        }}
      }}
    }}
  ]
}}
```

RELATIONSHIP ATTRIBUTES:
- Use `relationship_attributes` to group related properties (e.g., revenue_range, ebitda_multiple, etc.)
- Use snake_case for attribute names
- Attribute values should be strings (even for numbers)
- Group related metrics under a single relationship rather than creating multiple triples

EXAMPLE OUTPUT:
Given: "A Mid-Market business with revenue between ฿10M and ฿50M has a Bangkok EBITDA multiple of 5.5x–7.5x and a regional EBITDA multiple of 5.0x–7.0x in the Thai E-Commerce Market."

Output:
```json
{{
  "subject": {{"name": "Mid-Market Business", "type": "BusinessCategory"}},
  "predicate": "HAS_VALUATION_PROFILE",
  "object": {{"name": "Thai E-Commerce Market", "type": "Market"}},
  "relationship_attributes": {{
    "revenue_range": "฿10M–฿50M",
    "ebitda_multiple_bangkok": "5.5x–7.5x",
    "ebitda_multiple_regional": "5.0x–7.0x"
  }},
  "properties": {{
    "relationship_class": "VALUATION",
    "status": "Current",
    "evidence_quote": "A Mid-Market business with revenue between ฿10M and ฿50M has a Bangkok EBITDA multiple of 5.5x–7.5x",
    "validFrom": null,
    "validTo": null,
    "causal_link": null
  }}
}}
```

Extract all triples now. Remember: every number in the text must appear in at
least one triple's relationship_attributes. Group related attributes into single
relationships rather than creating separate triples for each attribute.

FORMAT PREFERENCE:
- Group related metrics (revenue, multiples, timelines) under one relationship
- Use relationship_attributes for specific values
- Keep predicates broad and meaningful

EXAMPLE COMPARISON:
OLD FORMAT (avoid):
  (Mid-Market Business)-[HAS_EBITDA_MULTIPLE_BANGKOK]->(5.5x–7.5x)
  (Mid-Market Business)-[HAS_EBITDA_MULTIPLE_REGIONAL]->(5.0x–7.0x)
  (Mid-Market Business)-[HAS_REVENUE_CEILING]->(฿50M)

NEW FORMAT (use):
  (Mid-Market Business)-[HAS_VALUATION_PROFILE {{
    revenue_range: "฿10M–฿50M",
    ebitda_multiple_bangkok: "5.5x–7.5x",
    ebitda_multiple_regional: "5.0x–7.0x"
  }}]->(Thai E-Commerce Market)

Aim for completeness with grouped relationships, not numerous granular triples."""

    return prompt
