"""Metadata extraction node for LangGraph workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.extract_metadata import extract_metadata, save_metadata

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def extract_metadata_node(state: "WorkflowState") -> "WorkflowState":
    """Extract metadata from the processed document using LLM."""
    try:
        print("📊 Extracting metadata...")

        if not state["input_file"]:
            raise ValueError("No input file available")

        path = Path(state["input_file"])
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {state['input_file']}")

        # Extract metadata using regex + CLI overrides
        metadata = extract_metadata(
            markdown_path=state["input_file"],
            location_moo_override=state.get("location_moo"),
            location_village_override=state.get("location_village"),
        )

        # Save metadata to Qdrant
        save_metadata(metadata)

        state["metadata"] = metadata
        state["current_step"] = "chunk_document"
        print(f"✅ Metadata extracted: {metadata.get('unique_id')}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Metadata extraction failed: {e}"
        print(f"❌ Error: {e}")

    return state
