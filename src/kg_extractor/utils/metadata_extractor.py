"""Metadata extraction using regex search + LLM fallback for remaining fields."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from kg_extractor.utils.parser import (
    get_openai_api_key,
    get_openrouter_api_key,
    get_api_key,
    get_google_api_key,
)
from kg_extractor.utils.prompts import create_metadata_extraction_prompt

_ALL_FIELDS = [
    "document_title",
    "document_file_name",
    "document_file_type",
    "document_total_pages",
    "document_content_type",
    "location_village",
    "location_moo",
    "location_country",
]


def _get_example_metadata(db_path: str) -> Optional[Dict[str, Any]]:
    """Fetch the first metadata record from database as an example.

    Args:
        db_path: Path to DuckDB database file

    Returns:
        Example metadata dictionary or None if database is empty
    """
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        result = conn.execute("SELECT * FROM metadata LIMIT 1").fetchone()
        conn.close()

        if result:
            columns = [
                'unique_id', 'document_title', 'document_file_name', 'document_file_type',
                'document_total_pages', 'document_content_type', 'location_village',
                'location_moo', 'location_country', 'created_at', 'updated_at'
            ]
            # Exclude timestamps from example
            example = dict(zip(columns, result))
            example.pop('created_at', None)
            example.pop('updated_at', None)
            example.pop('unique_id', None)
            return example
    except Exception as e:
        print(f"⚠️  Could not fetch example metadata: {e}")
    return None


def _search_metadata(content: str, markdown_path: str) -> Dict[str, Any]:
    """Extract metadata using regex/pattern search — no LLM cost.

    Handles predictable fields: file name, file type, total pages, and หมู่ number.
    """
    result: Dict[str, Any] = {}

    path = Path(markdown_path)

    # File name: strip "_analysis" suffix added by the pipeline
    result["document_file_name"] = path.stem.replace("_analysis", "")

    # File type: from markdown content marker or stem suffix pattern
    # The original file extension may be embedded in the stem (e.g., "report_pdf_analysis.md")
    stem = path.stem.replace("_analysis", "")
    for ext in ("pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "csv"):
        if stem.lower().endswith(f"_{ext}") or stem.lower().endswith(f".{ext}"):
            result["document_file_type"] = ext.upper()
            break

    # หมู่ number — very predictable pattern in Thai survey reports
    moo_match = re.search(r'หมู่(?:ที่)?\s*\d+', content)
    if moo_match:
        result["location_moo"] = moo_match.group(0).strip()

    # Total pages — look for common Thai/English page count patterns
    pages_match = re.search(
        r'(?:จำนวน\s*)?(\d+)\s*(?:หน้า\b|pages?\b)',
        content,
        re.IGNORECASE,
    )
    if pages_match:
        try:
            result["document_total_pages"] = int(pages_match.group(1))
        except ValueError:
            pass

    return result


def extract_metadata_with_llm(
    markdown_path: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    db_path: str = "metadata.db",
) -> Dict[str, Any]:
    """Extract metadata from analyzed markdown file using hybrid regex + LLM approach.

    First extracts predictable fields via regex search (zero LLM cost), then
    calls the LLM with a small content sample only for the remaining fields.

    Args:
        markdown_path: Path to the markdown analysis file
        llm_provider: LLM provider (openai, groq, nvidia, openrouter)
        llm_model: LLM model name
        db_path: Path to metadata database for fetching examples

    Returns:
        Dictionary containing extracted metadata
    """
    print(f"📊 Extracting metadata using {llm_provider}/{llm_model}...")

    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()

    source_file = Path(markdown_path).stem.replace('_analysis', '')

    # Step 1: fast regex search across full content
    searched = _search_metadata(content, markdown_path)

    # Step 2: determine which fields still need LLM
    missing_fields = [f for f in _ALL_FIELDS if f not in searched]

    # Step 3: call LLM only for missing fields, with a small content sample
    if missing_fields:
        # Fetch an example from the database to help LLM mimic format
        example = _get_example_metadata(db_path)

        # Sample: front of document (titles/headers) + tail (summary/location info)
        sample = content[:2000] + ("\n...\n" + content[-1000:] if len(content) > 3000 else "")
        prompt = create_metadata_extraction_prompt(sample, fields=missing_fields, example=example)

        if llm_provider == "openai":
            llm_metadata = _call_openai(prompt, llm_model)
        elif llm_provider == "groq":
            llm_metadata = _call_groq(prompt, llm_model)
        elif llm_provider == "nvidia":
            llm_metadata = _call_nvidia(prompt, llm_model)
        elif llm_provider == "openrouter":
            llm_metadata = _call_openrouter(prompt, llm_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    else:
        llm_metadata = {}

    # Merge: regex results take priority over LLM
    metadata = {**llm_metadata, **searched}

    # Generate unique ID from location fields
    moo = metadata.get("location_moo", "").replace("หมู่ที่ ", "").replace(" ", "")
    village = metadata.get("location_village", "").split("(")[0].strip()
    metadata["unique_id"] = f"{moo}_{village}" if moo and village else source_file

    llm_fields_used = missing_fields if missing_fields else []
    regex_fields_used = list(searched.keys())
    print(f"  🔍 Regex: {regex_fields_used}")
    print(f"  🤖 LLM:   {llm_fields_used}")
    print(f"✅ Metadata extracted: {metadata['unique_id']}")
    return metadata


def _call_openai(prompt: str, model: str) -> Dict[str, Any]:
    """Call OpenAI API for metadata extraction."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a metadata extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def _call_groq(prompt: str, model: str) -> Dict[str, Any]:
    """Call Groq API for metadata extraction."""
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a metadata extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def _call_nvidia(prompt: str, model: str) -> Dict[str, Any]:
    """Call NVIDIA API for metadata extraction."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = get_api_key()
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a metadata extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)


def _call_openrouter(prompt: str, model: str) -> Dict[str, Any]:
    """Call OpenRouter API for metadata extraction."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = get_openrouter_api_key()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a metadata extraction expert. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content
    # Extract JSON from markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)
