"""Metadata extraction node for LangGraph workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.extract_metadata import extract_metadata_with_llm, save_metadata

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def extract_metadata_node(state: "WorkflowState") -> "WorkflowState":
    """Extract metadata from the processed document using LLM."""
    try:
        print("📊 Extracting metadata...")

        if not state["markdown_path"]:
            raise ValueError("No markdown path available")

        path = Path(state["markdown_path"])
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {state['markdown_path']}")

        # Extract metadata using LLM
        metadata = extract_metadata_with_llm(
            markdown_path=state["markdown_path"],
            llm_provider=state["chunking_llm_provider"],  # Reuse chunking LLM config
            llm_model=state["chunking_llm_model"],
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
