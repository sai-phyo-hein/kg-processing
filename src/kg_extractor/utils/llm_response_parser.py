"""Utilities for parsing LLM responses with JSON content.

This module provides robust JSON extraction and repair utilities for parsing
LLM responses that may contain malformed JSON or extra text.
"""

import re
import json
from typing import Any, Dict, Optional


def extract_json_from_response(response: str) -> Optional[str]:
    """Extract JSON content from LLM response, handling code fences and prose.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Extracted JSON string, or None if no JSON found
    """
    if not response or not response.strip():
        return None
    
    text = response.strip()
    
    # Try fenced block first: ```json ... ``` or ``` ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    
    # Find outermost { ... } — handles leading/trailing prose
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    
    return text[start : end + 1]


def repair_json(text: str) -> str:
    """Best-effort repairs for common model JSON mistakes.
    
    Fixes:
      - Trailing commas before } or ]
      - Single-quoted strings
      - Unquoted keys
      - Duplicate opening braces  {{ → {
      - Stray newlines inside string values
      
    Args:
        text: JSON string that may contain errors
        
    Returns:
        Repaired JSON string
    """
    # Duplicate braces (f-string leak): {{ → {  and  }} → }
    text = re.sub(r"\{\{", "{", text)
    text = re.sub(r"\}\}", "}", text)

    # Trailing commas before closing brace/bracket
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Single-quoted strings → double-quoted (crude but catches simple cases)
    # Only replace when not inside an already double-quoted string
    text = re.sub(r"(?<![\\\"'])'([^']*)'(?![\\\"'])", r'"\1"', text)

    # Unquoted keys: word characters followed by colon not already quoted
    text = re.sub(r'(?<!")(\b\w+\b)\s*:', r'"\1":', text)

    # Collapse literal newlines inside string values (between quotes on same logical line)
    text = re.sub(r'(?<=\S)\n(?=\S)', ' ', text)

    return text


def parse_json_with_repair(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response with automatic extraction and repair.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Parsed JSON as dictionary, or None if parsing fails
    """
    # Extract JSON block
    text = extract_json_from_response(response)
    if not text:
        return None
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try with repair
    try:
        repaired = repair_json(text)
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def parse_triple_extraction_response(response: str) -> Dict[str, Any]:
    """Parse LLM response for triple extraction.
    
    Expected format:
    {
        "document_metadata": {
            "reference_date": str | null,
            "source_id": str | null,
            "chunk_id": int | null
        },
        "discovered_triples": [...]
    }
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with document_metadata and discovered_triples keys
    """
    empty = {
        "document_metadata": {
            "reference_date": None,
            "source_id": None,
            "chunk_id": None,
        },
        "discovered_triples": [],
    }

    if not response or not response.strip():
        return empty

    text = response.strip()

    # Reject known garbage / suppression responses
    GARBAGE_PATTERNS = [
        r"_PROCESSING_",
        r"_SUPPRESSED_",
        r"DISPLAY\}",
        r"^\s*\{[\s_A-Z]+\}",      # e.g. { _PROCESSING_ }
    ]
    for pat in GARBAGE_PATTERNS:
        if re.search(pat, text):
            print(f"Warning: Model returned a suppression/garbage response, skipping.")
            return empty

    # Extract and parse JSON
    data = parse_json_with_repair(response)
    if not data:
        print(f"Warning: No JSON object found in response.")
        print(f"Response content: {response[:300]}")
        return empty
    
    # Validate structure
    if "document_metadata" not in data or "discovered_triples" not in data:
        print(f"Warning: Invalid response structure — keys found: {list(data.keys())}")
        return empty
    
    return data


def parse_chunk_splits_response(response: str, section_line_count: int) -> list[int]:
    """Parse LLM response for chunk split-point extraction.
    
    Expected format:
    {
        "split_at": [1, 5, 12, 20, ...]
    }
    
    Args:
        response: LLM response text
        section_line_count: Total lines in section for validation
        
    Returns:
        Sorted list of 1-based line numbers where chunks begin
    """
    def _validate_splits(splits: list[int]) -> list[int]:
        """Deduplicate, sort, validate range, and ensure starts with 1."""
        valid = sorted(set(
            int(s) for s in splits
            if isinstance(s, (int, float)) and 1 <= int(s) <= section_line_count
        ))
        if not valid or valid[0] != 1:
            valid = [1] + [s for s in valid if s != 1]
        return valid

    # Extract JSON
    json_str = extract_json_from_response(response)
    if not json_str:
        print(f"⚠️  Could not extract JSON; treating section as one chunk.")
        return [1]
    
    # Try full JSON parse
    try:
        data = json.loads(json_str)
        if "split_at" in data and isinstance(data["split_at"], list):
            return _validate_splits(data["split_at"])
        # Handle old chunks format
        if "chunks" in data and isinstance(data["chunks"], list):
            n_chunks = len(data["chunks"])
            if n_chunks > 1:
                step = max(1, section_line_count // n_chunks)
                estimated = [1 + i * step for i in range(n_chunks)]
                print(f"⚠️  LLM returned content-format (chunks). "
                      f"Estimating {n_chunks} splits from chunk count.")
                return _validate_splits(estimated)
            return [1]
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract numbers from partial split_at array
    match = re.search(r'"split_at"\s*:\s*\[([0-9,\s]*)', json_str)
    if match:
        numbers_str = match.group(1)
        numbers = re.findall(r'\d+', numbers_str)
        if numbers:
            print(f"⚠️  Used regex fallback to extract {len(numbers)} split points.")
            return _validate_splits([int(n) for n in numbers])

    # Last resort: any array of numbers
    match = re.search(r'\[([0-9,\s]+)\]', json_str)
    if match:
        numbers = re.findall(r'\d+', match.group(1))
        if numbers:
            print(f"⚠️  Extracted split points from bare number array.")
            return _validate_splits([int(n) for n in numbers])

    print(f"⚠️  Could not parse response; treating section as one chunk.")
    return [1]
