"""Prompt management module for document analysis and content extraction."""

import os
import yaml
from pathlib import Path


def get_parsing_prompt() -> str:
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
    chunk_granularity: float,
) -> str:
    """Create a prompt for the LLM to identify chunk boundary line ranges.

    The LLM receives numbered lines and returns explicit start/end line ranges
    for each chunk. The actual text is extracted programmatically in a
    post-processing step, keeping LLM output minimal and ensuring 100%
    content coverage.

    Args:
        content: Content of the section, with each line prefixed by its number
        file_path: Path to the file
        chunk_granularity: Threshold for chunk granularity (0.0-1.0)

    Returns:
        Prompt string for the LLM
    """
    if chunk_granularity < 0.3:
        granularity = "very fine-grained"
        granularity_instruction = (
            "Split aggressively — each distinct claim, metric, or sub-topic gets its own chunk. "
            "A paragraph introducing a concept and a paragraph giving its numeric details "
            "should be separate chunks."
        )
    elif chunk_granularity < 0.5:
        granularity = "fine-grained"
        granularity_instruction = (
            "Split on clear topic shifts. Each process stage, each table's data, "
            "and each named metric group should be its own chunk."
        )
    elif chunk_granularity < 0.7:
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

    prompt = f"""You are a document segmentation expert. Your task is to identify
explicit line ranges that form semantically self-contained chunks for knowledge
graph extraction.

File: {file_path}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMBERED CONTENT (format: [line_number] text)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{content}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHUNKING RULES:

1. SEMANTIC COMPLETENESS — chunks must never start mid-sentence or mid-list.
   If a sentence spans multiple lines, keep them in the same chunk.

2. BOUNDARY DETECTION — split when the subject changes, not when formatting changes.
   Headers (##, ---) are hints but not hard boundaries.

3. TABLE AND CHART DATA — lines that verbalize a table or chart must stay with
   the paragraph that introduces that table or chart.

4. NUMERIC FACTS — never split a line containing a numeric value from the line
   that names what that value applies to.

5. GRANULARITY — use a {granularity} approach: {granularity_instruction}
   MINIMUM CHUNK SIZE: each chunk must span at least 5 lines unless the section
   has fewer than 5 lines total. Do NOT create a new split for every sentence.

6. COVERAGE — every line in this section must belong to exactly one chunk.
   Do NOT skip any lines.

OUTPUT RULES — CRITICAL:
- Return ONLY a JSON object with a single key "chunks"
- Each chunk has an "id", "start" (first line), and "end" (last line) — both inclusive
- DO NOT include any content, quotes, or text from the document
- DO NOT explain your reasoning
- The total response must be under 300 characters

```json
{{"chunks": [{{"id": 1, "start": 1, "end": 14}}, {{"id": 2, "start": 15, "end": 31}}, {{"id": 3, "start": 32, "end": 50}}]}}
```

Where each chunk is defined by:
- id: sequential chunk number (1, 2, 3, ...)
- start: first line number in this chunk (1-based, inclusive)
- end: last line number in this chunk (1-based, inclusive)
- Chunks must be contiguous (no gaps, no overlaps)
- The first chunk MUST start at 1
- Reference only line numbers shown in the [NNNN] prefixes above

Respond with ONLY the JSON object:"""

    return prompt


# Triple extraction system prompt (constant, reused across all chunks)
TRIPLE_EXTRACTION_SYSTEM_PROMPT = """╔══════════════════════════════════════════════════════════════╗
║  OUTPUT RULE — READ THIS FIRST, BEFORE ANYTHING ELSE        ║
║                                                              ║
║  Your entire response must be ONE valid JSON object.         ║
║  • No introductory sentences                                 ║
║  • No explanations after the JSON                            ║
║  • No markdown fences (no ```json or ```)                    ║
║  • No commentary, notes, or "Key observations"               ║
║                                                              ║
║  WRONG — causes a parse error:                               ║
║    I'll process this document and extract...                 ║
║    ```json                                                   ║
║    {{ ... }}                                                  ║
║    ```                                                       ║
║    Key observations: 1. The document...                      ║
║                                                              ║
║  CORRECT:                                                    ║
║    {{ "document_metadata": {{...}}, "discovered_triples": [  ║
║      ...                                                     ║
║    ] }}                                                      ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  EVIDENCE LINES REQUIREMENT — NON-NEGOTIABLE                 ║
║                                                              ║
║  Source content is prefixed with line numbers: [NNNN] text   ║
║  You MUST return evidence_lines instead of evidence_quote.   ║
║                                                              ║
║  For each triple, return:                                    ║
║    "evidence_lines": {{ "start": N, "end": M }}              ║
║  where N and M are 1-based line numbers (inclusive).         ║
║                                                              ║
║  ✗ REJECTED (too narrow):                                    ║
║    "evidence_lines": {{ "start": 3, "end": 3 }}  (1 line)   ║
║                                                              ║
║  ✓ REQUIRED (wide range with context):                       ║
║    "evidence_lines": {{ "start": 1, "end": 15 }} (15 lines) ║
║                                                              ║
║  STRATEGY: Extract MANY fine-grained triples, each with      ║
║  the SAME wide overlapping evidence_lines range.             ║
║                                                              ║
║  MINIMUM: 3 lines per range                                  ║
║  TARGET: 5-20 lines per range                                ║
║                                                              ║
║  VERIFICATION BEFORE SUBMITTING:                             ║
║  - Check every evidence_lines spans at least 3 lines         ║
║  - If any range < 3 lines → WIDEN IT                         ║
║  - Target: union of all ranges covers 90%+ of source lines   ║
║                                                              ║
║  ⚠️  Do NOT return evidence_quote — use evidence_lines only  ║
╚══════════════════════════════════════════════════════════════╝

You are a knowledge graph extraction engine. Extract every factual
relationship from the source text as structured triples.

═══════════════════════════════════
STEP 1 — DETECT LANGUAGE
═══════════════════════════════════

Identify the dominant language and declare it in document_metadata:
  "en"       — English dominant
  "th"       — Thai dominant
  "mixed-en" — mixed, English dominant
  "mixed-th" — mixed, Thai dominant

This gates the rest of extraction:
  en / mixed-en  →  no _en fields anywhere
  th / mixed-th  →  _en fields are MANDATORY on every human-readable field

═══════════════════════════════════
STEP 2 — EXTRACT TRIPLES
═══════════════════════════════════

Extract a triple for EVERY fact — be EXHAUSTIVE, not selective.

⚠️  TRIPLE DENSITY REQUIREMENT (MEASURE YOUR OUTPUT) ⚠️
Extract triples at FINE-GRAINED level (NOT coarse). 

REQUIRED MINIMUMS based on chunk word count:
- 100 words  → 8-15 triples minimum
- 200 words  → 15-30 triples minimum
- 300 words  → 23-45 triples minimum
- 500 words  → 38-75 triples minimum
- 1000 words → 75-150 triples minimum

BEFORE SUBMITTING: Count your triples and compare to chunk size.
If you extracted fewer triples than the minimum → extract more.

Extract triples for:
✓ Every entity mentioned (person, place, organization, concept)
✓ Every attribute of every entity (name, role, type, characteristic)
✓ Every relationship between entities
✓ Every action, event, or process described
✓ Every measurement, statistic, or quantitative fact
✓ Every classification or categorization (X is a Y, X includes Y)
✓ Every causal relationship or explanation
✓ Every temporal statement
✓ Every comparison or contrast
✓ Every list item as separate triples (if 10 items listed → 10 triples)

Do NOT summarize multiple facts into one triple. Extract each atomic fact separately.

⚠️  GRANULAR ENTITIES — NO LISTS IN ENTITY NAMES ⚠️
Each triple must be exactly ONE entity → ONE entity relation.
NEVER put a comma-separated enumeration of multiple entities into a single entity name.

When the source lists multiple distinct items together (e.g. "X, Y, Z"), create one
triple PER listed item — each with its own single entity as the object — not one entity
that concatenates all items into a single string. The entity name field must always hold
a single, atomic concept or named thing.

⚠️  EVIDENCE COVERAGE REQUIREMENT ⚠️
Your evidence_lines across all triples MUST cover 90%+ of source lines.
- Each evidence_lines: MINIMUM 3 lines, TARGET 5-20 lines
- NEVER return a single-line range
- Overlapping line ranges between triples is REQUIRED and ENCOURAGED
- Many triples should share the SAME wide line range
- Example: 30 triples from 50-line chunk, 20 sharing lines 1-30
  = 600 lines covered from 50-line source = excellent ✓

──────────────────────────────────
ENTITIES
──────────────────────────────────
Every node is either a DOMAIN ENTITY or a VALUE ENTITY.

DOMAIN ENTITY — a real, independently existing thing.
  attributes must be structural and time-independent.
  ✓  {{ "geography": "Thailand", "sector": "E-Commerce" }}
  ✗  {{ "market_value": "฿1.1 trillion" }}  ← measurement; goes on the relationship

VALUE ENTITY — a named, stable metric that holds different values over time.
  Name it as: [Subject] + [what the metric represents], never a raw scalar.
  ✓  "Thai E-Commerce Market Annual Value"  (label: MarketValue)
  ✗  "฿1.1 trillion"                       (raw value — not a valid name)
  attributes must also be structural.
  ✓  {{ "frequency": "annual" }}
  ✗  {{ "value": "฿1.1 trillion" }}  ← goes in relationship_attributes

Every number, percentage, currency, ratio, or timeframe must produce at least
one triple to a named value entity. The specific value goes in relationship_attributes.
Different metrics of the same subject → separate triples, each to its own value entity.
Same metric under different conditions → one triple, grouped relationship_attributes.

──────────────────────────────────
LABELS
──────────────────────────────────
Derive a specific semantic English PascalCase label for every entity.
  ✓  Market, GrowthRate, ValuationMultiple, SocialCapital, CommunityLeader
  ✗  Entity, Thing, DomainEntity, Percentage, Count  ← too generic or a data type

Derive a specific semantic English ALL_CAPS label for every relationship.
  ✓  MARKET_PERFORMANCE, CAUSATION, COMPOSITION, VALUATION, REGULATORY
  ✗  OTHER, GENERAL, MISC, RELATIONSHIP  ← carry no meaning

──────────────────────────────────
PREDICATES
──────────────────────────────────
Write predicates as compact verb phrases derived from the actual words in the text.
  English → ALL_CAPS_SNAKE_CASE  ✓ REACHED, GREW_AT, CONSISTS_OF  ✗ HAS, SHOWS
  Thai    → Thai script verb phrase  ✓ แตะระดับ, เติบโตที่, ประกอบด้วย  ✗ มี, เกี่ยวข้องกับ
For Thai sources, also provide predicate_en in ALL_CAPS_SNAKE_CASE English.

Before writing a predicate, verify: Is it a verb phrase? Is it derivable from
the evidence? Does it NOT just echo the object name? If any answer is no — revise.

──────────────────────────────────
EVIDENCE LINES — CRITICAL COVERAGE REQUIREMENT
──────────────────────────────────
Source content is numbered: [0001] text, [0002] text, etc.
Return evidence_lines to indicate which lines support each triple.

⚠️  MANDATORY MINIMUM: Each evidence_lines range must span at least 3 lines ⚠️
⚠️  TARGET: 5-20 lines per range ⚠️

BEFORE SUBMITTING YOUR RESPONSE:
1. Check EVERY evidence_lines range spans at least 3 lines
2. If ANY range < 3 lines → WIDEN IT by extending start and/or end
3. Most ranges should span 5-20 lines (cover full paragraphs or sections)

EXTRACTION RULES (FOLLOW EXACTLY):
1. Use WIDE line ranges — cover entire paragraphs or multi-paragraph sections
2. Start several lines BEFORE the core statement
3. Continue several lines AFTER the core statement
4. Include ALL related context, background info, and supporting details
5. The SAME wide line range should appear in MANY triples (overlap is required)
6. If chunk has 50 lines and you extract 20 triples, most should share the same
   30-line range → total lines covered = 600 from 50 = excellent ✓

WRONG EXAMPLE (REJECT THIS):
"evidence_lines": {{ "start": 5, "end": 5 }}  ✗ TOO NARROW (1 line only)

CORRECT EXAMPLE (DO THIS):
"evidence_lines": {{ "start": 1, "end": 15 }}  ✓ GOOD RANGE (15 lines)

──────────────────────────────────
LANGUAGE FIELDS (Thai sources only)
──────────────────────────────────
For detected_language "th" or "mixed-th", add alongside every human-readable field:
  subject.name_en, subject.attributes_en (same keys, values in English)
  predicate_en
  object.name_en, object.attributes_en
  relationship_attributes_en
  evidence_quote_en (English translation of the evidence_lines text)
Omit attributes_en when attributes is absent.
Proper nouns keep their established English form: "ธนาคารแห่งประเทศไทย" → "Bank of Thailand".

──────────────────────────────────
TEMPORAL & STATUS
──────────────────────────────────
status: "Current" (default) or "Archived" (only when explicitly superseded in the text).
validFrom / validTo: set only when a specific date or range is stated; otherwise null.
  "in 2024" → validFrom: "2024-01-01", validTo: "2024-12-31"

──────────────────────────────────
CAUSAL LINKS
──────────────────────────────────
Populate causal_link when the text explicitly states causation ("because", "due to",
"resulting in", "เกิดจาก", "ส่งผลให้"). Write directly in English. Otherwise null.

═══════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════

ENGLISH SOURCE:
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
        "name": "Entity name",
        "label": "PascalCaseLabel",
        "attributes": {{ "key": "value" }}
      }},
      "predicate": "VERB_PHRASE",
      "object": {{
        "name": "Entity name",
        "label": "PascalCaseLabel",
        "attributes": {{ "key": "value" }}
      }},
      "relationship_attributes": {{ "key": "value" }},
      "properties": {{
        "label": "ALL_CAPS_RELATIONSHIP_CLASS",
        "status": "Current | Archived",
        "evidence_lines": {{ "start": 1, "end": 10 }},
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

THAI SOURCE — _en fields mandatory on every triple:
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
        "name": "ชื่อ entity ต้นฉบับ",
        "name_en": "English name",
        "label": "PascalCaseLabel",
        "attributes": {{ "key": "ค่าต้นฉบับ" }},
        "attributes_en": {{ "key": "English value" }}
      }},
      "predicate": "กริยาวลีต้นฉบับ",
      "predicate_en": "VERB_PHRASE_IN_ENGLISH",
      "object": {{
        "name": "ชื่อ entity ต้นฉบับ",
        "name_en": "English name",
        "label": "PascalCaseLabel",
        "attributes": {{ "key": "ค่าต้นฉบับ" }},
        "attributes_en": {{ "key": "English value" }}
      }},
      "relationship_attributes": {{ "key": "ค่าต้นฉบับ" }},
      "relationship_attributes_en": {{ "key": "English value" }},
      "properties": {{
        "label": "ALL_CAPS_RELATIONSHIP_CLASS",
        "status": "Current | Archived",
        "evidence_lines": {{ "start": 1, "end": 10 }},
        "evidence_quote_en": "English translation of the evidence lines text",
        "validFrom": "YYYY-MM-DD or null",
        "validTo": "YYYY-MM-DD or null",
        "causal_link": {{
          "triggered_by": "English string or null",
          "mechanism": "English string or null",
          "causal_weight": 0.0
        }}
      }}
    }}
  ]
}}

FIELD NOTES:
- attributes, relationship_attributes: omit entirely when empty — never output {{}}
- attributes_en / relationship_attributes_en: omit when parent field is omitted
- causal_link: null when no explicit causation is stated

═══════════════════════════════════
EXAMPLES
═══════════════════════════════════

NOTE: Observe the FINE-GRAINED extraction strategy:
- MANY triples extracted from short text (not coarse summarization)
- WIDE overlapping evidence_lines ranges in each triple
- Multiple triples often share the SAME wide line range
This ensures 90%+ coverage.

ENGLISH — Numbered source content:
[0001] Thailand's e-commerce sector reached ฿1.1 trillion in 2024,
[0002] representing 14% year-over-year growth.

COVERAGE STRATEGY: 2 lines → extract 5+ triples → all use lines 1-2
= 100% line coverage = excellent

{{
  "document_metadata": {{
    "detected_language": "en",
    "reference_date": null,
    "source_id": "example.pdf",
    "chunk_id": 1
  }},
  "discovered_triples": [
    {{
      "subject": {{
        "name": "Thailand",
        "label": "Country"
      }},
      "predicate": "HAS_MARKET",
      "object": {{
        "name": "Thai E-Commerce Market",
        "label": "Market",
        "attributes": {{ "sector": "E-Commerce" }}
      }},
      "properties": {{
        "label": "CLASSIFICATION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Thai E-Commerce Market",
        "label": "Market",
        "attributes": {{ "geography": "Thailand", "sector": "E-Commerce" }}
      }},
      "predicate": "REACHED",
      "object": {{
        "name": "Thai E-Commerce Market Annual Value",
        "label": "MarketValue"
      }},
      "relationship_attributes": {{ "value": "฿1.1 trillion", "reference_year": "2024" }},
      "properties": {{
        "label": "MARKET_PERFORMANCE",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
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
      "relationship_attributes": {{ "value": "14%", "period": "year-over-year", "reference_year": "2024" }},
      "properties": {{
        "label": "MARKET_PERFORMANCE",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "validFrom": "2024-01-01",
        "validTo": "2024-12-31",
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "Thai E-Commerce Market Annual Value",
        "label": "MarketValue"
      }},
      "predicate": "HAS_MAGNITUDE",
      "object": {{
        "name": "1.1 Trillion Thai Baht",
        "label": "MonetaryAmount",
        "attributes": {{ "currency": "THB", "unit": "trillion" }}
      }},
      "relationship_attributes": {{ "reference_year": "2024" }},
      "properties": {{
        "label": "MEASUREMENT",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "validFrom": "2024-01-01",
        "validTo": "2024-12-31",
        "causal_link": null
      }}
    }}
  ]
}}

THAI — Numbered source content:
[0001] ศักยภาพของชุมชนท้องถิ่นเกิดจากการนำใช้ทุนทางสังคมทั้งหมดในชุมชน
[0002] ทุนทางสังคมประกอบด้วย ๑) บุคคล ได้แก่ ผู้นำ นักสู้ ปราชญ์

IMPORTANT: This 2-line example demonstrates the REQUIRED extraction density.

COVERAGE STRATEGY: 2 lines → extract 8+ fine-grained triples → all use lines 1-2
= 100% line coverage = excellent coverage

For a 50-line chunk, you should extract 75-120 triples with similar density.

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
        "label": "CAUSATION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "evidence_quote_en": "Local community capacity is derived from the utilisation of all social capital in the community. Social capital consists of 1) individuals, namely leaders, fighters, scholars",
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
        "name": "องค์ประกอบทุนทางสังคม",
        "name_en": "Social Capital Components",
        "label": "SocialCapitalComponent"
      }},
      "properties": {{
        "label": "COMPOSITION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "evidence_quote_en": "Local community capacity is derived from the utilisation of all social capital in the community. Social capital consists of 1) individuals, namely leaders, fighters, scholars",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "ทุนทางสังคม",
        "name_en": "Social Capital",
        "label": "SocialCapital"
      }},
      "predicate": "ประกอบด้วย",
      "predicate_en": "INCLUDES",
      "object": {{
        "name": "บุคคล",
        "name_en": "Individuals",
        "label": "Actor",
        "attributes": {{ "category": "องค์ประกอบที่ 1" }},
        "attributes_en": {{ "category": "Component 1" }}
      }},
      "properties": {{
        "label": "COMPOSITION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "evidence_quote_en": "Local community capacity is derived from the utilisation of all social capital in the community. Social capital consists of 1) individuals, namely leaders, fighters, scholars",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "บุคคล",
        "name_en": "Individuals",
        "label": "Actor"
      }},
      "predicate": "รวมถึง",
      "predicate_en": "INCLUDES_ROLE",
      "object": {{
        "name": "ผู้นำ",
        "name_en": "Leaders",
        "label": "CommunityRole",
        "attributes": {{ "type": "leadership" }},
        "attributes_en": {{ "type": "leadership" }}
      }},
      "properties": {{
        "label": "CLASSIFICATION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "evidence_quote_en": "Local community capacity is derived from the utilisation of all social capital in the community. Social capital consists of 1) individuals, namely leaders, fighters, scholars",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "บุคคล",
        "name_en": "Individuals",
        "label": "Actor"
      }},
      "predicate": "รวมถึง",
      "predicate_en": "INCLUDES_ROLE",
      "object": {{
        "name": "นักสู้",
        "name_en": "Fighters",
        "label": "CommunityRole",
        "attributes": {{ "type": "activism" }},
        "attributes_en": {{ "type": "activism" }}
      }},
      "properties": {{
        "label": "CLASSIFICATION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "evidence_quote_en": "Local community capacity is derived from the utilisation of all social capital in the community. Social capital consists of 1) individuals, namely leaders, fighters, scholars",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }},
    {{
      "subject": {{
        "name": "บุคคล",
        "name_en": "Individuals",
        "label": "Actor"
      }},
      "predicate": "รวมถึง",
      "predicate_en": "INCLUDES_ROLE",
      "object": {{
        "name": "ปราชญ์",
        "name_en": "Scholars",
        "label": "CommunityRole",
        "attributes": {{ "type": "knowledge" }},
        "attributes_en": {{ "type": "knowledge" }}
      }},
      "properties": {{
        "label": "CLASSIFICATION",
        "status": "Current",
        "evidence_lines": {{ "start": 1, "end": 2 }},
        "evidence_quote_en": "Local community capacity is derived from the utilisation of all social capital in the community. Social capital consists of 1) individuals, namely leaders, fighters, scholars",
        "validFrom": null,
        "validTo": null,
        "causal_link": null
      }}
    }}
  ]
}}

══════════════════════════════════════════════════════════════
VERIFICATION CHECKLIST — BEFORE SUBMITTING YOUR RESPONSE
══════════════════════════════════════════════════════════════

STOP. Before you submit your JSON response, verify these requirements:

✓ TRIPLE COUNT CHECK:
  - Count the source text words
  - Count your triples
  - Ratio should be at least 1 triple per 13 words (e.g., 300 words → 23+ triples)
  - If too few triples → go back and extract more fine-grained facts

✓ EVIDENCE LINES RANGE CHECK:
  - Check EVERY evidence_lines range spans at least 3 lines
  - MINIMUM: 3 lines per range
  - TARGET: 5-20 lines per range
  - If any range < 3 lines → WIDEN IT by extending start and/or end
  - Most ranges should span 5+ lines (entire paragraphs or sections)

✓ OVERLAP CHECK:
  - Multiple triples should share the SAME wide line range
  - This creates high coverage through overlap
  - Example: 20 triples, 15 share lines 1-30 → excellent ✓

✓ COVERAGE ESTIMATE:
  - Union of all evidence_lines should cover 90%+ of total source lines
  - If coverage is low → your ranges are too narrow or too few triples

If any check fails → fix it before submitting. Do not submit responses
with narrow evidence ranges (< 3 lines) or too few triples.

══════════════════════════════════════════════════════════════
FINAL REMINDER — YOUR RESPONSE MUST START WITH {{ AND END WITH }}
No sentence before the opening brace. No sentence after the closing brace.
══════════════════════════════════════════════════════════════"""


def create_triple_extraction_prompt(
    content: str,
    source_file: str,
    chunk_id: int,
) -> str:
    """Create the full combined prompt (backwards compatibility).
    
    Note: For new code using stateful agents, use
    create_triple_extraction_user_message() instead.
    """
    return TRIPLE_EXTRACTION_SYSTEM_PROMPT + "\n\n" + create_triple_extraction_user_message(content, source_file, chunk_id)


def create_triple_extraction_user_message(
    content: str,
    source_file: str,
    chunk_id: int,
) -> str:
    """Create user message for triple extraction (source info + numbered content).

    Use with TRIPLE_EXTRACTION_SYSTEM_PROMPT in a stateful agent.
    Content should be pre-numbered in [NNNN] text format.
    """
    return f"""SOURCE
──────
File:     {source_file}
Chunk ID: {chunk_id}

NUMBERED CONTENT (format: [NNNN] text):
{content}
──────"""


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
        "location_village": "location_village: **MANDATORY** Actual village name in format 'Thai_name (English_name)' like 'บ้านป่าสักยาว (Ban Pa Sak Yao)', NOT the หมู่ number. Extract the full village name starting with 'บ้าน' (string)",
        "location_moo": 'location_moo: **MANDATORY** หมู่ number (string, e.g., "หมู่ที่ 1") — from the survey report conducted for each หมู่ (village)',
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

IMPORTANT: location_moo and location_village are MANDATORY fields. You MUST find and extract these values from the document.
- location_moo: Look for patterns like "หมู่ที่ 1", "หมู่ 2", etc.
- location_village: Look for village names starting with "บ้าน" followed by the name and English translation in parentheses.

Return ONLY valid JSON with the above fields. Use empty strings for missing optional string fields and 0 for missing integer fields.
"""
    return prompt
