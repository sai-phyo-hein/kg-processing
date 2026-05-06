"""Metadata extraction using regex search + CLI overrides only."""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

def _load_metadata_config() -> dict:
    """Load metadata field config from metadata_config.yaml."""
    config_path = Path(__file__).parent.parent.parent.parent.parent.parent / "configs" / "metadata_config.yaml"
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata_config.yaml: {e}")


_METADATA_CONFIG = _load_metadata_config()
_MANDATORY_FIELDS: list = _METADATA_CONFIG.get("mandatory") or []


def extract_metadata(
    markdown_path: str,
    location_moo_override: Optional[str] = None,
    location_village_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract metadata using only CLI-provided values (no regex search).

    Args:
        markdown_path: Path to the markdown analysis file (kept for signature compatibility)
        location_moo_override: location_moo value from CLI
        location_village_override: location_village value from CLI

    Returns:
        Dictionary containing extracted metadata
    """
    print(f"📊 Extracting metadata using CLI values only...")

    source_file = Path(markdown_path).stem.replace('_analysis', '')

    # Use only CLI-provided values
    metadata = {}
    if location_moo_override:
        metadata["location_moo"] = location_moo_override
    if location_village_override:
        metadata["location_village"] = location_village_override

    # Validate mandatory fields
    for field in _MANDATORY_FIELDS:
        if not str(metadata.get(field, "")).strip():
            raise ValueError(f"{field} is mandatory but not provided: {source_file}")

    # Construct unique_id from mandatory fields (joined with "_"), fall back to file name
    if _MANDATORY_FIELDS:
        metadata["unique_id"] = "_".join(
            str(metadata[f]).strip() for f in _MANDATORY_FIELDS
        )
    else:
        metadata["unique_id"] = source_file

    cli_values = []
    if location_moo_override:
        cli_values.append("location_moo")
    if location_village_override:
        cli_values.append("location_village")
    
    print(f"  🎯 CLI values: {cli_values}")
    print(f"✅ Metadata extracted: {metadata['unique_id']}")
    return metadata
