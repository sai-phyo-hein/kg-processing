"""Triple extraction node for LangGraph workflow."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.extract_triples import extract_triples_from_chunks

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def _load_chunks_from_manifest(chunks_path: str) -> list[dict]:
    """Load chunks from the new directory-based format (manifest.json + files).

    Also supports the legacy single-JSON format for backward compatibility.

    Args:
        chunks_path: Path to either a manifest.json inside a chunks directory,
                     or a legacy single chunks JSON file.

    Returns:
        List of chunk dicts with at least ``chunk_id`` and ``content``.
    """
    path = Path(chunks_path)

    # --- New format: manifest.json inside a chunks directory ---
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            path = manifest_path
        else:
            raise FileNotFoundError(
                f"No manifest.json found in chunks directory: {chunks_path}"
            )

    # path now points to either manifest.json or a legacy single JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks = data.get("chunks", [])
    parent_dir = path.parent

    # If chunks have a "file" key, load content from individual files
    if raw_chunks and "file" in raw_chunks[0]:
        chunks = []
        for entry in raw_chunks:
            chunk_file = parent_dir / entry["file"]
            if chunk_file.exists():
                content = chunk_file.read_text(encoding="utf-8")
            else:
                content = ""
            chunks.append({
                "chunk_id": entry.get("chunk_id", 0),
                "content": content,
            })
        return chunks

    # Legacy: chunks already contain content inline
    return raw_chunks


def extract_triples_node(state: "WorkflowState") -> "WorkflowState":
    """Extract knowledge graph triples from chunks."""
    try:
        print("🔗 Extracting triples...")

        if not state["chunks_path"]:
            raise ValueError("No chunks path available")

        chunks_path = Path(state["chunks_path"])
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found at: {state['chunks_path']}")

        # Load chunks (supports both new directory and legacy single-file formats)
        chunks = _load_chunks_from_manifest(state["chunks_path"])
        print(f"📄 Found {len(chunks)} chunks for triple extraction")

        # Get community_id from metadata
        community_id = None
        if state["metadata"]:
            community_id = state["metadata"].get("unique_id")

        # Extract triples
        output_path = extract_triples_from_chunks(
            chunks=chunks,
            source_file=state["input_file"],
            llm_provider=state["triplet_llm_provider"],
            llm_model=state["triplet_llm_model"],
            output_dir=state["output_dir"],
            community_id=community_id,
        )

        state["triples_path"] = output_path
        state["current_step"] = "refine_triples" if state["refine_triples"] else "build_graph"
        print(f"✅ Triple extraction completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Triple extraction failed: {e}"
        print(f"❌ Error: {e}")

    return state
