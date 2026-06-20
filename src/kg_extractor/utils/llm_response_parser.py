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
        "discovered_triples": [...]
    }

    Args:
        response: LLM response text

    Returns:
        Dictionary with a discovered_triples key
    """
    empty = {
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
        # Check if JSON appears truncated (more opening than closing braces)
        open_braces = response.count("{")
        close_braces = response.count("}")
        if open_braces > close_braces:
            print(f"⚠️  JSON parsing failed: Response appears truncated ({open_braces} opening vs {close_braces} closing braces). "
                  f"Check max_tokens setting or LLM output limits.")
        else:
            print(f"⚠️  JSON parsing failed: Malformed response structure.")
        return empty
    
    # Validate structure
    if "discovered_triples" not in data:
        print(f"⚠️  Invalid response structure — expected 'discovered_triples', found: {list(data.keys())}")
        return empty

    # Validate evidence_lines ranges in discovered_triples
    for triple in data.get("discovered_triples", []):
        props = triple.get("properties", {})
        evidence_lines = props.get("evidence_lines")
        if evidence_lines and isinstance(evidence_lines, dict):
            start = evidence_lines.get("start")
            end = evidence_lines.get("end")
            if start is not None and end is not None:
                try:
                    evidence_lines["start"] = max(1, int(start))
                    evidence_lines["end"] = max(evidence_lines["start"], int(end))
                except (ValueError, TypeError):
                    # Remove invalid evidence_lines
                    props.pop("evidence_lines", None)

    return data


def parse_chunk_ranges_response(response: str, section_line_count: int) -> list[dict]:
    """Parse LLM response for explicit chunk line ranges.

    Expected format:
    {
        "chunks": [
            {"id": 1, "start": 1, "end": 14},
            {"id": 2, "start": 15, "end": 31}
        ]
    }

    Also accepts the legacy "split_at" format and converts it to ranges.

    Args:
        response: LLM response text
        section_line_count: Total lines in section for validation

    Returns:
        List of {"id": int, "start": int, "end": int} (1-based, inclusive).
        Empty list on failure.
    """
    def _validate_ranges(ranges: list[dict]) -> list[dict]:
        """Validate, deduplicate, and sort chunk ranges."""
        valid = []
        for r in ranges:
            start = int(r.get("start", 0))
            end = int(r.get("end", 0))
            if start < 1 or end < start or start > section_line_count:
                continue
            end = min(end, section_line_count)
            valid.append({"id": int(r.get("id", len(valid) + 1)), "start": start, "end": end})
        # Sort by start line
        valid.sort(key=lambda x: x["start"])
        # Ensure first chunk starts at 1
        if valid and valid[0]["start"] != 1:
            valid.insert(0, {"id": 0, "start": 1, "end": valid[0]["start"] - 1})
        # Fill any gaps by extending the previous chunk
        for i in range(1, len(valid)):
            if valid[i]["start"] > valid[i - 1]["end"] + 1:
                valid[i - 1]["end"] = valid[i]["start"] - 1
        # Re-number ids
        for i, r in enumerate(valid):
            r["id"] = i + 1
        return valid

    # Extract JSON
    json_str = extract_json_from_response(response)
    if not json_str:
        print("⚠️  Could not extract JSON for chunk ranges.")
        return []

    # Try full JSON parse
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try repair
        try:
            data = json.loads(repair_json(json_str))
        except json.JSONDecodeError:
            pass
        else:
            return _parse_ranges_from_data(data, section_line_count, _validate_ranges)
        # Regex fallback: extract chunk objects
        return _regex_fallback_ranges(json_str, section_line_count, _validate_ranges)

    return _parse_ranges_from_data(data, section_line_count, _validate_ranges)


def _parse_ranges_from_data(
    data: dict, section_line_count: int, validator
) -> list[dict]:
    """Parse chunk ranges from a parsed JSON dict (new or legacy format)."""
    if not isinstance(data, dict):
        return []

    # New format: {"chunks": [{"id": 1, "start": 1, "end": 10}, ...]}
    if "chunks" in data and isinstance(data["chunks"], list):
        raw = []
        for c in data["chunks"]:
            if isinstance(c, dict) and "start" in c and "end" in c:
                raw.append(c)
        if raw:
            return validator(raw)

    # Legacy format: {"split_at": [1, 15, 32]}
    if "split_at" in data and isinstance(data["split_at"], list):
        splits = sorted(set(
            int(s) for s in data["split_at"]
            if isinstance(s, (int, float)) and 1 <= int(s) <= section_line_count
        ))
        if not splits or splits[0] != 1:
            splits = [1] + [s for s in splits if s != 1]
        # Convert split points to ranges
        ranges = []
        for i, sp in enumerate(splits):
            end = splits[i + 1] - 1 if i + 1 < len(splits) else section_line_count
            ranges.append({"id": i + 1, "start": sp, "end": end})
        print("⚠️  LLM returned legacy split_at format; converted to chunk ranges.")
        return ranges

    return []


def _regex_fallback_ranges(
    json_str: str, section_line_count: int, validator
) -> list[dict]:
    """Last-resort regex extraction of chunk ranges from partial JSON."""
    # Try to find individual {"id": N, "start": N, "end": N} objects
    pattern = r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"start"\s*:\s*(\d+)\s*,\s*"end"\s*:\s*(\d+)\s*\}'
    matches = re.findall(pattern, json_str)
    if matches:
        raw = [
            {"id": int(m[0]), "start": int(m[1]), "end": int(m[2])}
            for m in matches
        ]
        print(f"⚠️  Regex fallback extracted {len(raw)} chunk ranges.")
        return validator(raw)

    # Try to find bare number arrays (legacy split_at fallback)
    match = re.search(r'\[([0-9,\s]+)\]', json_str)
    if match:
        numbers = [int(n) for n in re.findall(r'\d+', match.group(1))]
        if numbers:
            splits = sorted(set(n for n in numbers if 1 <= n <= section_line_count))
            if not splits or splits[0] != 1:
                splits = [1] + [s for s in splits if s != 1]
            ranges = []
            for i, sp in enumerate(splits):
                end = splits[i + 1] - 1 if i + 1 < len(splits) else section_line_count
                ranges.append({"id": i + 1, "start": sp, "end": end})
            print("⚠️  Regex fallback extracted split points, converted to ranges.")
            return ranges

    print("⚠️  Could not parse chunk ranges from response.")
    return []


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
