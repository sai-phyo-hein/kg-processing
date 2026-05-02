"""Prompt management module for document analysis and content extraction."""

import os
import yaml
from pathlib import Path


def get_schema_content() -> str:
    """Read and return the schema content from schema.yaml.

    Returns:
        Schema content as a formatted string for use in prompts
    """
    # Get the directory containing schema.yaml
    current_dir = Path(__file__).resolve()
    schema_path = current_dir.parent / "schema.yaml"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_data = yaml.safe_load(f)

    # Format the schema content for use in prompts
    formatted_schema = """
═══════════════════════════════════════
SCHEMA VALIDATION RULES
═══════════════════════════════════════

You MUST extract triples that conform to the RECAP schema. Only use the following
node types and relations:

VALID NODE TYPES:
"""

    # Add node types
    for node_type in schema_data['node_types']:
        formatted_schema += f"- {node_type}\n"

    # Add relations
    formatted_schema += "\nVALID RELATIONS:\n"
    for relation in schema_data['relations']:
        formatted_schema += f"- {relation}\n"

    # Add enum values
    formatted_schema += "\nENUM VALUES:\n"
    for enum_name, values in schema_data['enum_values'].items():
        formatted_schema += f"- {enum_name}: {', '.join(values)}\n"

    formatted_schema += """
CRITICAL: Do NOT extract triples with node types or relations that are not in the
above lists. If the text describes entities or relationships that don't match the
schema, either:
1. Map them to the closest valid schema type/relation, OR
2. Skip that triple entirely

"""
    return formatted_schema


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
SECTION 0 — OUTPUT LANGUAGE AND ENTITY PRESERVATION
═══════════════════════════════════════════════════════

Detect the dominant language of the document from the body text on this page.
Write ALL output — including verbalized table sentences and verbalized chart
sentences — in that same language.

If the document is in Thai, produce Thai output.
If the document is in Japanese, produce Japanese output.
If the document is in French, produce French output.
English is used ONLY when the document itself is in English.

ENTITY PRESERVATION: Regardless of the document's dominant language, always
reproduce the following exactly as they appear in the source — never translate
or paraphrase them:
- Proper nouns and named entities (people, companies, places)
- Brand names and product names
- Ticker symbols and financial identifiers
- Technical terms that appear in a non-dominant script

PRONOUN REPLACEMENT: When extracting body text, replace all pronouns that refer
to a named entity with the entity's name or a specific descriptor. This ensures
each sentence is unambiguous when read in isolation.

Examples:
  Source:  "Acme Corp reported strong earnings. They attributed this to exports."
  Output:  "Acme Corp reported strong earnings. Acme Corp attributed this to exports."

  Source:  "The CEO spoke at the conference. He said growth would continue."
  Output:  "The CEO spoke at the conference. The CEO said growth would continue."

  Source:  "Revenue rose 14%. It was driven by the Bangkok market."
  Output:  "Revenue rose 14%. The 14% revenue increase was driven by the Bangkok market."

Do NOT replace pronouns that refer to concepts with no clear named antecedent,
or where the antecedent is within the same sentence.

═══════════════════════════════════════════════════════
SECTION 1 — TEXT CONTENT (HIGHEST PRIORITY)
═══════════════════════════════════════════════════════

- Extract ALL body text. This is your PRIMARY task.
- Preserve all facts, figures, and phrasing exactly as written. This fidelity
  applies to content — not to PDF rendering artefacts. Fix broken line-wraps
  and hyphenation artefacts introduced by PDF rendering.
- Do NOT paraphrase, summarise, or omit any sentence.
- Preserve all figures, percentages, currency values, and named entities exactly
  as they appear (e.g. "฿1.1 trillion", "14%", "2.5x–5x"). Named entity
  preservation rules are defined in Section 0.
- Apply pronoun replacement as defined in Section 0.
- If a sentence is incomplete due to a page break, write it as-is and note
  "[continues on next page]" at the end.
- Include ALL paragraphs, bullet points, and narrative text.
- Never skip text content to focus on tables or charts.

═══════════════════════════════════════════════════════
SECTION 2 — TABLES
═══════════════════════════════════════════════════════

Tables contain the highest-density factual data. You MUST convert every table
into one declarative sentence per data row. Never produce JSON. Never summarise.

Note: "verbatim" in these rules means cell values — numbers, units, ranges, and
qualifiers — must be reproduced exactly. It does not mean sentence structure is
fixed; construct natural sentences in the document's dominant language.

RULES:
1. Begin with an anchor sentence naming the table, written in the document's
   dominant language (as identified in Section 0):
   English:  "According to [Table Title], the following data applies."
   Thai:     "ตามตาราง [Table Title] ข้อมูลมีดังนี้"
   (Use the equivalent phrasing in whatever language the document is written in.)

2. Write ONE sentence per row in the document's dominant language:
   "[Row subject] has [col2_header] of [col2_value], [col3_header] of
   [col3_value], and [col4_header] of [col4_value]."

3. Every cell value must appear verbatim — including units, ranges, and
   qualifiers (e.g. "3.5x–5.0x", "฿10M–฿50M", "75–120 days").

4. If a column contains a range (min–max), write the full range, not just one end.

5. If a cell is empty or merged, write "not specified" (or its equivalent in the
   document's dominant language) for that field.

EXAMPLE — given this table:
┌──────────────────┬───────────────┬──────────────────┬──────────────────┐
│ Business Size    │ Revenue Range │ EBITDA (Bangkok) │ EBITDA (Regional)│
├──────────────────┼───────────────┼──────────────────┼──────────────────┤
│ Small E-Commerce │ < ฿10M        │ 3.5x – 5.0x      │ 3.0x – 4.5x      │
│ Mid-Market       │ ฿10M – ฿50M   │ 5.5x – 7.5x      │ 5.0x – 7.0x      │
└──────────────────┴───────────────┴──────────────────┴──────────────────┘

CORRECT OUTPUT (English document):
"According to Table 1: Revenue-Based Valuation Multiples for Thai E-Commerce
Agencies (2025), the following data applies. A Small E-Commerce business with
revenue below ฿10M has a Bangkok EBITDA multiple of 3.5x–5.0x and a regional
EBITDA multiple of 3.0x–4.5x. A Mid-Market business with revenue between
฿10M and ฿50M has a Bangkok EBITDA multiple of 5.5x–7.5x and a regional
EBITDA multiple of 5.0x–7.0x."

CORRECT OUTPUT (Thai document):
"ตามตารางที่ 1: ตัวคูณมูลค่าตามรายได้สำหรับเอเจนซี่อีคอมเมิร์ซไทย (2025)
ข้อมูลมีดังนี้ ธุรกิจอีคอมเมิร์ซขนาดเล็กที่มีรายได้ต่ำกว่า ฿10M มีตัวคูณ
EBITDA ในกรุงเทพฯ อยู่ที่ 3.5x–5.0x และตัวคูณ EBITDA ในต่างจังหวัดอยู่ที่
3.0x–4.5x ธุรกิจระดับกลางที่มีรายได้ระหว่าง ฿10M ถึง ฿50M มีตัวคูณ EBITDA
ในกรุงเทพฯ อยู่ที่ 5.5x–7.5x และตัวคูณ EBITDA ในต่างจังหวัดอยู่ที่ 5.0x–7.0x"

WRONG OUTPUT (never do this):
- Producing a JSON object with the table data
- Writing "The table shows higher multiples for larger businesses"
- Omitting any row or any cell value
- Writing table sentences in English when the document is in another language

═══════════════════════════════════════════════════════
SECTION 3 — CHARTS AND DIAGRAMS
═══════════════════════════════════════════════════════

Charts encode quantitative facts as visual positions. Your job is to read each
labeled data point and write it as a sentence in the document's dominant
language. Never describe visual appearance.

RULES:
1. Begin with an anchor sentence naming the chart, written in the document's
   dominant language (as identified in Section 0):
   English: "The chart titled '[Chart Title]' shows [metric] for [subject(s)]."
   (Use the equivalent phrasing in whatever language the document is written in.)

2. For each labeled data point or bar, write one sentence in the document's
   dominant language:
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

CORRECT OUTPUT (English document):
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
- Writing chart sentences in English when the document is in another language.

═══════════════════════════════════════════════════════
SECTION 4 — CONTENT FILTERING
═══════════════════════════════════════════════════════

SKIP entirely (output nothing for these):
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

All output must be in the document's dominant language (as identified in
Section 0). Do not switch to English for table or chart verbalization.

Your output should be a single continuous text block containing:
- All extracted text from the page (FIRST and MOST IMPORTANT), with pronoun
  replacement applied as defined in Section 0
- Verbalized table sentences (one sentence per table row)
- Verbalized chart sentences (one sentence per data point)

All content should be combined into one continuous prose block in document order.

CRITICAL: Output ONLY the text content. Do not include any JSON structure,
metadata fields, or formatting markers. Just the plain text that can be
concatenated with other pages.

IMPORTANT: Never cut off text content mid-sentence. If you're running out of
space, complete the current sentence first, then stop."""


def create_chunking_prompt(
    content: str,
    file_path: str,
    similarity_threshold: float,
) -> str:
    """Create a prompt for the LLM to analyze content and determine chunk boundaries.

    Args:
        content: Full content of the markdown file
        file_path: Path to the file
        similarity_threshold: Threshold for chunk granularity (0.0-1.0)

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

PRIORITY: Content coverage (Rule 6) > Semantic completeness (Rule 1)

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

5. GRANULARITY — use a {granularity} approach: {granularity_instruction}

6. COMPREHENSIVE PROCESSING — process ALL content in the section.
   Do NOT skip or discard any content unless it's purely whitespace.
   Every sentence, table reference, and data point must be included in chunks.
   Even repetitive content, headers, and structural elements should be preserved.
   CRITICAL: Your chunks must collectively contain at least 90% of the original content.
   Content preservation takes priority over all other rules.

7. PRONOUN REPLACEMENT — replace ALL pronouns with explicit references.
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
      "content": "The full verbatim text of this chunk..."
    }},
    {{
      "chunk_id": 2,
      "content": "The full verbatim text of this chunk..."
    }}
  ]
}}
```

The `content` field must contain the actual text — not a summary or placeholder.

CRITICAL VALIDATION REQUIREMENT:
Before returning your JSON, verify that:
1. The total character count of all chunk contents is at least 90% of the original content character count
2. No substantive content has been skipped or summarized
3. All sentences, data points, and factual information are preserved

If your chunks contain less than 90% of the original content, revise your chunking strategy
to include more content rather than splitting differently.

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
        schema_rules = get_schema_content()

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
STEP 0 — LANGUAGE DETECTION (MANDATORY FIRST STEP)
═══════════════════════════════════════

Before extracting any triples, you MUST identify the dominant language of the
source content and declare it in document_metadata as detected_language.

  detected_language: "en"   — English dominant
  detected_language: "th"   — Thai dominant
  detected_language: "mixed-en"  — mixed, English dominant
  detected_language: "mixed-th"  — mixed, Thai dominant

This declaration is not optional. It gates the entire extraction:
  - If detected_language is "en" or "mixed-en" → no _en fields anywhere
  - If detected_language is "th" or "mixed-th" → _en fields are MANDATORY
    on every triple, on every human-readable field

═══════════════════════════════════════
LANGUAGE RULES — APPLY TO EVERY FIELD
═══════════════════════════════════════

  Entity names         → source language
  Predicate            → source language
  Attribute values     → source language
  Attribute keys       → always English snake_case        (programmatic)
  label                → always English PascalCase        (programmatic)
  relationship_class   → always English ALL_CAPS          (programmatic)
  evidence_quote       → always verbatim from source      (never translated)

For non-English sources, add these translation fields alongside every
human-readable field:

  subject.name          → subject.name_en
  subject.attributes    → subject.attributes_en
      (same keys, values translated to English)
  predicate             → predicate_en  (ALL_CAPS_SNAKE_CASE English verb phrase)
  object.name           → object.name_en
  object.attributes     → object.attributes_en
      (same keys, values translated to English)
  relationship_attributes       → relationship_attributes_en
      (same keys, values translated to English)

Fields that never get translated:
  label, relationship_class, attribute keys, evidence_quote, validFrom, validTo

Translation quality:
  - Translate meaning accurately, not word-for-word
  - Preserve proper nouns where an established English form exists
    ("ธนาคารแห่งประเทศไทย" → "Bank of Thailand")
  - Numeric values that are language-neutral ("14%", "5.5x–7.5x") may be kept
    as-is in _en fields; translate only surrounding descriptive text
  - causal_link fields write directly in English — no _en suffix needed

═══════════════════════════════════════
EXTRACTION RULES
═══════════════════════════════════════

──────────────────────────────────────
RULE 1 — EXHAUSTIVE EXTRACTION
──────────────────────────────────────
Extract a triple for EVERY fact in the text. Do not select only the "most
important" ones. A chunk of 200 words should typically yield 10–25 triples.
If you find fewer than 8, re-read the text and look for facts you missed.

──────────────────────────────────────
RULE 2 — LABELS ARE DYNAMIC
──────────────────────────────────────
Do not use a fixed list of labels. For every entity, derive the most
semantically accurate English PascalCase label that describes what the entity
IS or what KIND OF METRIC it represents.

PRINCIPLES FOR CHOOSING A LABEL:
  - Be specific enough to be meaningful and queryable
  - Reflect the domain of the source content — financial, social, legal,
    scientific, community, environmental, or any other domain
  - For domain entities: the label names the KIND OF THING
      Actor, Organization, Community, SocialCapital, Resource, Concept,
      LegalRequirement, ProcessStage, Location, Strategy, RiskFactor,
      CommunityLeader, Policy, Institution — or any other accurate term
  - For value entities: the label names the KIND OF METRIC
      MarketValue, GrowthRate, ValuationMultiple, ProcessTimeline, MarketSize,
      CommunityCapacity, ParticipationRate, FundingAmount, RiskLevel,
      CompletionRate — or any other accurate term

FORBIDDEN LABELS — never use these regardless of domain:
  DomainEntity, ValueEntity, Entity, Thing, Object, Item, Node, Other,
  Percentage, CurrencyValue, Count, Ratio, Number, Float
  ↑ these are too generic to be meaningful or are raw data types

──────────────────────────────────────
RULE 3 — RELATIONSHIP CLASS IS DYNAMIC
──────────────────────────────────────
Do not use a fixed list of relationship classes. For every predicate, derive
the most semantically accurate English ALL_CAPS category that describes the
conceptual nature of the relationship.

PRINCIPLES FOR CHOOSING A RELATIONSHIP CLASS:
  - Describe what KIND OF CONNECTION exists between subject and object
  - Be broad enough to group similar relationships, specific enough to be
    meaningful
  - Draw from any conceptual category that fits the domain:
      COMPOSITION      — subject contains or is made up of object
      CAUSATION        — subject causes, produces, or enables object
      CLASSIFICATION   — subject is a type or instance of object
      QUANTIFICATION   — subject is measured by or reaches a value
      ASSOCIATION      — subject is linked to or associated with object
      TEMPORAL         — subject precedes, follows, or occurs during object
      SPATIAL          — subject is located in, near, or part of object
      VALUATION        — subject is priced at or valued relative to object
      MARKET_POSITION  — subject ranks or competes relative to object
      MARKET_PERFORMANCE — subject achieves a measured market outcome
      MARKET_REACH     — subject covers or serves a population or area
      PROCESS_TIMELINE — subject takes a duration to complete
      TRANSACTION_OUTCOME — subject achieves a deal or transaction result
      REGULATORY       — subject is governed or constrained by object
      FINANCIAL_PERFORMANCE — subject achieves a financial result
      RISK             — subject is exposed to or characterised by a risk

  If none of the above fits, invent a new ALL_CAPS term that accurately
  describes the relationship. Never use: OTHER, GENERAL, MISC, UNKNOWN,
  RELATIONSHIP, CONNECTION — these carry no meaning.

──────────────────────────────────────
RULE 4 — TWO KINDS OF ENTITIES
──────────────────────────────────────
Every node is either a DOMAIN ENTITY or a VALUE ENTITY.

DOMAIN ENTITIES
A real, independently existing thing in the domain. Node attributes must be
GENERAL and TIME-INDEPENDENT — structural characteristics that do not change
with time or measurement:
  Good: {{ "geography": "Thailand", "sector": "E-Commerce",
           "founding_type": "community-based" }}
  Bad:  {{ "market_value": "฿1.1 trillion", "member_count": "240" }}
        ↑ time-specific measurements — belong on the relationship, not the node

VALUE ENTITIES
A named, stable metric or measure that can hold different specific values over
time. Name and label describe WHAT the metric is — not its current value.

  Naming value entities:
    name = [Subject Name] + [what the value represents], in the source language
    name_en = same pattern translated to English (for non-English sources)

    Good: "มูลค่าตลาด E-Commerce ไทยรายปี" / "Thai E-Commerce Market Annual Value"
          "ศักยภาพชุมชนท้องถิ่น" / "Local Community Capacity"
          "EBITDA Multiple ของธุรกิจขนาดกลาง" / "Mid-Market Business EBITDA Multiple"
    Bad:  "฿1.1 ล้านล้านบาท" / "14%" ← raw values used as names
          "EBITDA Multiple"            ← too generic, not tied to a subject

  Node attributes for value entities must also be general and time-independent:
    Good: {{ "frequency": "annual", "metric_type": "transaction_multiple" }}
    Bad:  {{ "value": "฿1.1 trillion", "reference_year": "2024" }}
          ↑ specific readings — always go in relationship_attributes

──────────────────────────────────────
RULE 5 — EVERY SCALAR FACT PRODUCES A TRIPLE
──────────────────────────────────────
Every number, percentage, currency amount, ratio, count, or timeframe in the
text must produce at least one triple — domain entity → predicate → named value
entity — with the specific value and any temporal context in
relationship_attributes.

When multiple scalar values describe the SAME metric under different conditions,
group them into ONE triple with multiple keys in relationship_attributes.

When scalar values describe DIFFERENT metrics of the same subject, produce
SEPARATE triples — one per metric — each pointing to its own named value entity.

  Same metric, different conditions → ONE triple, grouped relationship_attributes:
    "trades at Bangkok EBITDA multiple of 5.5x–7.5x and regional of 5.0x–7.0x"
    → one triple to "Mid-Market Business EBITDA Multiple" (ValuationMultiple)
      relationship_attributes: {{
        ebitda_multiple_bangkok: "5.5x–7.5x",
        ebitda_multiple_regional: "5.0x–7.0x"
      }}

  Different metrics, same subject → SEPARATE triples:
    "reached ฿1.1 trillion in 2024, representing 14% year-over-year growth"
    → triple 1 to "Thai E-Commerce Market Annual Value" (MarketValue)
    → triple 2 to "Thai E-Commerce Market Growth Rate" (GrowthRate)

  Ordinal/ranking values → triple to a value entity with appropriate label:
    "positioning as ASEAN's second-largest digital marketplace"
    → triple to "Thai E-Commerce Market ASEAN Rank" (MarketRank)
      relationship_attributes: {{ rank: "2", scope: "ASEAN",
                                  basis: "digital marketplace size" }}

──────────────────────────────────────
RULE 6 — PREDICATES ARE VERB PHRASES GROUNDED IN THE EVIDENCE
──────────────────────────────────────
Write all predicates as compact verb phrases in the source language, derived
from the actual language used in the source text.

  English → ALL_CAPS_SNAKE_CASE verb phrase
    Good: REACHED, GREW_AT, SERVES, IS_VALUED_AT, CLOSES_FASTER_THAN,
          REQUIRES_APPROVAL_FROM, CONSISTS_OF, IS_DERIVED_FROM, COMPLETES_IN
    Bad:  HAS, SHOWS, DEMONSTRATES, IMPACTS, RELATES_TO
    Bad:  HAS_ONLINE_SHOPPERS, HAS_EBITDA_MULTIPLE ← noun phrases not verbs

  Thai → compact verb phrase in Thai script
    Good: แตะระดับ, เติบโตที่, ให้บริการ, ซื้อขายที่, เกิดจาก, ประกอบด้วย,
          ถูกประเมินมูลค่าที่, ปิดดีลเร็วกว่า, ต้องได้รับอนุมัติจาก, ใช้เวลา
    Bad:  มี, แสดง, เกี่ยวข้องกับ

Before writing a predicate, verify all three:
  1. Is it a verb phrase — not a noun phrase?
  2. Is it derivable from the actual language of the evidence quote?
  3. Does it NOT echo or restate the name of the object?
If any answer is no — revise the predicate or restructure the triple.

──────────────────────────────────────
RULE 7 — VALID OBJECT ENTITIES
──────────────────────────────────────
The object of a triple must be either:
  a) A real domain entity that exists independently in the domain, OR
  b) A named value entity following the naming convention in Rule 4

The object must NOT be:
  - A raw scalar value used directly as a name ("14%", "฿1.1 trillion")
  - A metric label that echoes the predicate
  - An alias or restatement of the subject (self-loop)
  - An entity implied by the predicate but not present in the text

──────────────────────────────────────
RULE 8 — ENTITY NAMING CONSISTENCY
──────────────────────────────────────
The same real-world entity must have exactly one name across all triples in
this chunk. name is the sole node identifier — inconsistent naming creates
duplicate nodes in the graph. This applies to both domain and value entities.

──────────────────────────────────────
RULE 9 — TEMPORAL STATUS
──────────────────────────────────────
Set status to:
  "Current"  — most recent known state; no date given, or date is the most
               recent data point with no newer figure mentioned
  "Archived" — explicitly situated in a past period AND more recent data is
               expected or mentioned elsewhere in the text

Default to "Current". Do not mark Archived simply because a year is mentioned.

Set validFrom / validTo only when a specific date or year range is stated:
  "in 2024"      → validFrom: "2024-01-01", validTo: "2024-12-31"
  "as of Q1 2023" → validFrom: "2023-01-01", validTo: "2023-03-31"
  No date stated  → validFrom: null, validTo: null

──────────────────────────────────────
RULE 10 — EVIDENCE QUOTE
──────────────────────────────────────
evidence_quote must be a verbatim substring from the content — the shortest
phrase that directly supports this triple. Never fabricate or paraphrase. The
predicate must be derivable from the language in this quote. evidence_quote is
never translated — it always stays in the source language.

──────────────────────────────────────
RULE 11 — CAUSAL LINKS (optional but encouraged)
──────────────────────────────────────
If the text explicitly states that X caused or enabled Y — using words like
"because", "due to", "resulting in", "enabling", "driven by", "เกิดจาก",
"เนื่องจาก", "ส่งผลให้" — populate causal_link fields directly in English.
Otherwise set all causal fields to null.

═══════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════

Return ONLY valid JSON. No preamble, no explanation, no markdown outside the JSON.

ENGLISH-DOMINANT SOURCE (detected_language: "en" or "mixed-en"):
```json
{{
  "document_metadata": {{
    "detected_language": "en",
    "reference_date": "YYYY-MM-DD or null",
    "source_id": "{source_file}",
    "chunk_id": {chunk_id}
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "Entity name in English",
        "label": "SemanticPascalCaseLabel",
        "attributes": {{
          "snake_case_key": "value"
        }}
      }},
      "predicate": "VERB_PHRASE_IN_ENGLISH",
      "object": {{
        "name": "Entity name in English",
        "label": "SemanticPascalCaseLabel",
        "attributes": {{
          "snake_case_key": "value"
        }}
      }},
      "relationship_attributes": {{
        "snake_case_key": "value"
      }},
      "properties": {{
        "relationship_class": "SEMANTIC_ALL_CAPS_CLASS",
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

NON-ENGLISH-DOMINANT SOURCE (detected_language: "th" or "mixed-th"):
_en fields are MANDATORY on every triple — omitting them is an error.
```json
{{
  "document_metadata": {{
    "detected_language": "th",
    "reference_date": "YYYY-MM-DD or null",
    "source_id": "{source_file}",
    "chunk_id": {chunk_id}
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "ชื่อ entity ในภาษาต้นฉบับ",
        "name_en": "Entity name translated to English",
        "label": "SemanticPascalCaseLabel",
        "attributes": {{
          "snake_case_key": "ค่าในภาษาต้นฉบับ"
        }},
        "attributes_en": {{
          "snake_case_key": "value translated to English"
        }}
      }},
      "predicate": "กริยาวลีในภาษาต้นฉบับ",
      "predicate_en": "VERB_PHRASE_IN_ENGLISH",
      "object": {{
        "name": "ชื่อ entity ในภาษาต้นฉบับ",
        "name_en": "Entity name translated to English",
        "label": "SemanticPascalCaseLabel",
        "attributes": {{
          "snake_case_key": "ค่าในภาษาต้นฉบับ"
        }},
        "attributes_en": {{
          "snake_case_key": "value translated to English"
        }}
      }},
      "relationship_attributes": {{
        "snake_case_key": "ค่าในภาษาต้นฉบับ"
      }},
      "relationship_attributes_en": {{
        "snake_case_key": "value translated to English"
      }},
      "properties": {{
        "relationship_class": "SEMANTIC_ALL_CAPS_CLASS",
        "status": "Current | Archived",
        "evidence_quote": "ข้อความต้นฉบับ verbatim — ไม่แปล",
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

FIELD NOTES:
- subject.attributes / object.attributes: omit entirely when empty — never output {{}}
- subject.attributes_en / object.attributes_en: omit if parent attributes is omitted
- relationship_attributes: omit entirely when empty — never output {{}}
- relationship_attributes_en: omit if relationship_attributes is omitted
- label: semantic English PascalCase — specific to the domain and entity type;
         never DomainEntity, Entity, Thing, Object, Other, or raw data types
- relationship_class: semantic English ALL_CAPS — specific to the nature of the
         relationship; never OTHER, GENERAL, MISC, UNKNOWN, RELATIONSHIP
- predicate_en: ALL_CAPS_SNAKE_CASE English verb phrase — mandatory for non-English
- evidence_quote: verbatim from source — never translated, never paraphrased

═══════════════════════════════════════
EXAMPLES
═══════════════════════════════════════

ENGLISH SOURCE EXAMPLE (detected_language: "en")
Given: "Thailand's e-commerce sector reached ฿1.1 trillion in 2024, representing
14% year-over-year growth and positioning as ASEAN's second-largest digital
marketplace, with over 40 million online shoppers and 7,400 registered
e-commerce businesses. A Mid-Market business with revenue between ฿10M and ฿50M
trades at a Bangkok EBITDA multiple of 5.5x–7.5x and a regional multiple of
5.0x–7.0x."

```json
{{
  "document_metadata": {{
    "detected_language": "en",
    "reference_date": null,
    "source_id": "example_en.pdf",
    "chunk_id": 1
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "Thai E-Commerce Market",
        "label": "Market",
        "attributes": {{
          "geography": "Thailand",
          "sector": "E-Commerce"
        }}
      }},
      "predicate": "REACHED",
      "object": {{
        "name": "Thai E-Commerce Market Annual Value",
        "label": "MarketValue"
      }},
      "relationship_attributes": {{
        "value": "฿1.1 trillion",
        "reference_year": "2024"
      }},
      "properties": {{
        "relationship_class": "MARKET_PERFORMANCE",
        "status": "Current",
        "evidence_quote": "Thailand's e-commerce sector reached ฿1.1 trillion in 2024",
        "validFrom": "2024-01-01",
        "validTo": "2024-12-31",
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Thai E-Commerce Market",
        "label": "Market"
      }},
      "predicate": "GREW_AT",
      "object": {{
        "name": "Thai E-Commerce Market Growth Rate",
        "label": "GrowthRate"
      }},
      "relationship_attributes": {{
        "value": "14%",
        "period": "year-over-year",
        "reference_year": "2024"
      }},
      "properties": {{
        "relationship_class": "MARKET_PERFORMANCE",
        "status": "Current",
        "evidence_quote": "representing 14% year-over-year growth",
        "validFrom": "2024-01-01",
        "validTo": "2024-12-31",
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Thai E-Commerce Market",
        "label": "Market"
      }},
      "predicate": "RANKS_IN",
      "object": {{
        "name": "Thai E-Commerce Market ASEAN Rank",
        "label": "MarketRank"
      }},
      "relationship_attributes": {{
        "rank": "2",
        "scope": "ASEAN",
        "basis": "digital marketplace size"
      }},
      "properties": {{
        "relationship_class": "MARKET_POSITION",
        "status": "Current",
        "evidence_quote": "positioning as ASEAN's second-largest digital marketplace",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Thai E-Commerce Market",
        "label": "Market"
      }},
      "predicate": "SERVES",
      "object": {{
        "name": "Thai E-Commerce Market Online Shopper Count",
        "label": "MarketSize"
      }},
      "relationship_attributes": {{
        "value": "40 million"
      }},
      "properties": {{
        "relationship_class": "MARKET_REACH",
        "status": "Current",
        "evidence_quote": "with over 40 million online shoppers",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Thai E-Commerce Market",
        "label": "Market"
      }},
      "predicate": "CONTAINS",
      "object": {{
        "name": "Thai E-Commerce Market Registered Business Count",
        "label": "MarketSize"
      }},
      "relationship_attributes": {{
        "value": "7,400"
      }},
      "properties": {{
        "relationship_class": "MARKET_REACH",
        "status": "Current",
        "evidence_quote": "7,400 registered e-commerce businesses",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Mid-Market Business",
        "label": "BusinessCategory"
      }},
      "predicate": "IS_VALUED_AT",
      "object": {{
        "name": "Mid-Market Business EBITDA Multiple",
        "label": "ValuationMultiple",
        "attributes": {{
          "metric_type": "transaction_multiple"
        }}
      }},
      "relationship_attributes": {{
        "ebitda_multiple_bangkok": "5.5x–7.5x",
        "ebitda_multiple_regional": "5.0x–7.0x",
        "applicable_revenue_range": "฿10M–฿50M"
      }},
      "properties": {{
        "relationship_class": "VALUATION",
        "status": "Current",
        "evidence_quote": "trades at a Bangkok EBITDA multiple of 5.5x–7.5x and a regional multiple of 5.0x–7.0x",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }}
  ]
}}
```

THAI SOURCE EXAMPLE (detected_language: "th") — _en fields mandatory
Given (Thai): "ศักยภาพของชุมชนท้องถิ่นเกิดจากการนำใช้ทุนทางสังคมทั้งหมดในชุมชน
ทุนทางสังคมประกอบด้วย ๑) บุคคล ได้แก่ ผู้นำ นักสู้ ปราชญ์
ตลาด E-Commerce ไทยแตะระดับ ฿1.1 ล้านล้านบาท ในปี 2567 เติบโต 14% เมื่อเทียบปีต่อปี"

```json
{{
  "document_metadata": {{
    "detected_language": "th",
    "reference_date": null,
    "source_id": "example_th.pdf",
    "chunk_id": 1
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "ศักยภาพของชุมชนท้องถิ่น",
        "name_en": "Local Community Capacity",
        "label": "CommunityCapacity"
      }},
      "predicate": "เกิดจาก",
      "predicate_en": "IS_DERIVED_FROM",
      "object": {{
        "name": "ทุนทางสังคมในชุมชน",
        "name_en": "Community Social Capital",
        "label": "SocialCapital"
      }},
      "properties": {{
        "relationship_class": "CAUSATION",
        "status": "Current",
        "evidence_quote": "ศักยภาพของชุมชนท้องถิ่นเกิดจากการนำใช้ทุนทางสังคมทั้งหมดในชุมชน",
        "validFrom": null,
        "validTo": null,
        "causal_link": {{
          "triggered_by": "utilisation of all social capital in the community",
          "mechanism": "mobilisation of collective community resources",
          "causal_weight": 0.9
        }}
      }}
    }},
    {{
      "subject": {{
        "name": "ทุนทางสังคม",
        "name_en": "Social Capital",
        "label": "SocialCapital"
      }},
      "predicate": "ประกอบด้วย",
      "predicate_en": "CONSISTS_OF",
      "object": {{
        "name": "บุคคล",
        "name_en": "Individuals",
        "label": "Actor",
        "attributes": {{
          "roles": "ผู้นำ นักสู้ ปราชญ์"
        }},
        "attributes_en": {{
          "roles": "leaders, fighters, scholars"
        }}
      }},
      "properties": {{
        "relationship_class": "COMPOSITION",
        "status": "Current",
        "evidence_quote": "ประกอบด้วย ๑) บุคคล ได้แก่ ผู้นำ นักสู้ ปราชญ์",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "ตลาด E-Commerce ไทย",
        "name_en": "Thai E-Commerce Market",
        "label": "Market",
        "attributes": {{
          "geography": "ประเทศไทย",
          "sector": "E-Commerce"
        }},
        "attributes_en": {{
          "geography": "Thailand",
          "sector": "E-Commerce"
        }}
      }},
      "predicate": "แตะระดับ",
      "predicate_en": "REACHED",
      "object": {{
        "name": "มูลค่าตลาด E-Commerce ไทยรายปี",
        "name_en": "Thai E-Commerce Market Annual Value",
        "label": "MarketValue"
      }},
      "relationship_attributes": {{
        "value": "฿1.1 ล้านล้านบาท",
        "reference_year": "2567"
      }},
      "relationship_attributes_en": {{
        "value": "฿1.1 trillion",
        "reference_year": "2024"
      }},
      "properties": {{
        "relationship_class": "MARKET_PERFORMANCE",
        "status": "Current",
        "evidence_quote": "ตลาด E-Commerce ไทยแตะระดับ ฿1.1 ล้านล้านบาท ในปี 2567",
        "validFrom": "2024-01-01",
        "validTo": "2024-12-31",
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "ตลาด E-Commerce ไทย",
        "name_en": "Thai E-Commerce Market",
        "label": "Market"
      }},
      "predicate": "เติบโตที่",
      "predicate_en": "GREW_AT",
      "object": {{
        "name": "อัตราการเติบโตของตลาด E-Commerce ไทย",
        "name_en": "Thai E-Commerce Market Growth Rate",
        "label": "GrowthRate"
      }},
      "relationship_attributes": {{
        "value": "14%",
        "period": "เมื่อเทียบปีต่อปี",
        "reference_year": "2567"
      }},
      "relationship_attributes_en": {{
        "value": "14%",
        "period": "year-over-year",
        "reference_year": "2024"
      }},
      "properties": {{
        "relationship_class": "MARKET_PERFORMANCE",
        "status": "Current",
        "evidence_quote": "เติบโต 14% เมื่อเทียบปีต่อปี",
        "validFrom": "2024-01-01",
        "validTo": "2024-12-31",
        "causal_link": null
      }}
    }}
  ]
}}
```

═══════════════════════════════════════
ANTI-PATTERN REFERENCE
═══════════════════════════════════════

  ✗ Generic label used:
    label: "DomainEntity" / "Entity" / "Thing" / "Other"
    Fix: derive a specific semantic label — "SocialCapital", "Community",
         "Actor", "Market", "CommunityLeader" — whatever the entity actually is

  ✗ Raw data type used as label:
    label: "Percentage" / "CurrencyValue" / "Count"
    Fix: derive a label describing the kind of metric — "GrowthRate", "MarketValue"

  ✗ OTHER used as relationship_class:
    relationship_class: "OTHER" / "GENERAL" / "MISC"
    Fix: derive a specific class — "CAUSATION", "COMPOSITION", "ASSOCIATION" —
         or invent an accurate ALL_CAPS term

  ✗ _en fields missing on non-English source:
    detected_language: "th" → triple has no name_en, no predicate_en
    Fix: _en fields are MANDATORY for all non-English sources — omitting them
         is an extraction error

  ✗ Raw value used as entity name:
    object: {{ "name": "฿1.1 ล้านล้านบาท", "label": "CurrencyValue" }}
    Fix: name the metric — "มูลค่าตลาด E-Commerce ไทยรายปี" (MarketValue)

  ✗ Predicate not grounded in evidence:
    predicate: มีผู้ซื้อ  (no such verb in evidence)
    Fix: derive verb from text; if no verb connects subject to a second entity,
         absorb the fact as a node attribute or a value entity triple

  ✗ Self-loop:
    (ชุมชนท้องถิ่น)-[เป็น]->(ชุมชน)
    Fix: one canonical name; never alias the subject as the object

  ✗ Grouping different metrics into one triple:
    relationship_attributes: {{ value: "฿1.1 ล้านล้านบาท", growth_rate: "14%" }}
    Fix: separate triples, each to its own named value entity

  ✗ attributes_en present when attributes is absent:
    "attributes_en": {{ "roles": "leaders" }}  (no attributes field present)
    Fix: omit attributes_en entirely when attributes is omitted

  ✗ evidence_quote translated:
    "evidence_quote": "Local community capacity is derived from social capital"
    Fix: evidence_quote is always verbatim source text — never translated

  ✗ Time-specific values as node attributes:
    subject: {{ "attributes": {{ "market_value": "฿1.1 trillion" }} }}
    Fix: node attributes are structural; measurements go on the relationship

  ✗ Empty objects:
    "relationship_attributes": {{}}  /  "attributes": {{}}
    Fix: omit the key entirely

═══════════════════════════════════════
FINAL CHECKLIST — RUN BEFORE OUTPUT
═══════════════════════════════════════

  □ detected_language declared in document_metadata before any extraction
  □ _en fields present on EVERY triple for non-English sources — no exceptions
  □ _en fields absent for English-dominant sources
  □ label is a specific semantic English PascalCase term — not generic, not a data type
  □ relationship_class is a specific semantic English ALL_CAPS term — not OTHER/GENERAL
  □ Predicate is a verb phrase in source language, grounded in the evidence quote
  □ predicate_en is ALL_CAPS_SNAKE_CASE English verb phrase (non-English sources only)
  □ Different metrics produce separate triples to separate value entities
  □ Same metric under different conditions grouped into one triple
  □ Node attributes are structural and time-independent
  □ All measurements and temporal context are in relationship_attributes
  □ status defaults to "Current" unless fact is explicitly superseded
  □ No raw values as entity names; no generic or data-type labels
  □ No empty attribute or relationship_attributes objects
  □ No self-loops, phantom objects, or predicate echoing the object name
  □ evidence_quote is verbatim from source — never translated

Extract all triples now."""

    return prompt


def create_metadata_extraction_prompt(content: str, fields: list[str] | None = None, example: dict[str, Any] | None = None) -> str:
    """Create a prompt for extracting document metadata.

    Args:
        content: Document content sample to analyze
        fields: List of field names to extract. If None, extracts all fields.
        example: Example metadata from database to help LLM mimic the format

    Returns:
        Prompt string for metadata extraction
    """
    all_field_descriptions = {
        "document_title": 'document_title: Title of the document (string)',
        "document_file_name": 'document_file_name: Original file name (string)',
        "document_file_type": 'document_file_type: File type (string, e.g., "PDF", "DOCX")',
        "document_total_pages": "document_total_pages: Total number of pages (integer)",
        "document_content_type": 'document_content_type: Content type (string, e.g., "mixed", "text", "table")',
        "location_village": "location_village: Actual village name like 'บ้านป่าสักยาว' (Ban Pa Sak Yao), NOT the หมู่ number. Extract the full village name starting with 'บ้าน' (string)",
        "location_moo": 'location_moo: หมู่ number (string, e.g., "หมู่ที่ 1") — from the survey report conducted for each หมู่ (village)',
        "location_country": "location_country: Country name (string) — from the survey report conducted for each หมู่ (village)",
    }

    if fields is None:
        fields = list(all_field_descriptions.keys())

    field_lines = "\n".join(f"- {all_field_descriptions[f]}" for f in fields if f in all_field_descriptions)

    # Add example section if provided
    example_section = ""
    if example:
        import json
        example_filtered = {k: v for k, v in example.items() if k in fields}
        example_section = f"""
Example format from existing data (mimic this style):
{json.dumps(example_filtered, ensure_ascii=False, indent=2)}

"""

    prompt = f"""Analyze the following document excerpt and extract metadata in JSON format with these exact fields:

Required fields:
{field_lines}

{example_section}Document content:
{content}

Return ONLY valid JSON with the above fields. Use empty strings for missing string fields and 0 for missing integer fields.
"""
    return prompt
