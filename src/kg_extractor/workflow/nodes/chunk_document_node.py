"""Semantic chunking node for LangGraph workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.chunk_document import chunk_markdown_file

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def chunk_document_node(state: "WorkflowState") -> "WorkflowState":
    """Perform semantic chunking on the processed document."""
    try:
        print("🧩 Performing semantic chunking...")

        if not state["markdown_path"]:
            raise ValueError("No markdown path available")

        path = Path(state["markdown_path"])
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {state['markdown_path']}")

        # community_id (metadata unique_id) keys the S3 upload of chunk files.
        community_id = None
        if state["metadata"]:
            community_id = state["metadata"].get("unique_id")

        # Perform chunking (uploads chunks to S3 when community_id is set)
        output_path = chunk_markdown_file(
            file_path=state["markdown_path"],
            chunk_granularity=state["chunk_granularity"],
            llm_provider=state["chunking_llm_provider"],
            llm_model=state["chunking_llm_model"],
            output_dir=state["output_dir"],
            community_id=community_id,
        )

        state["chunks_path"] = output_path
        state["current_step"] = "extract_triples"
        print(f"✅ Chunking completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Chunking failed: {e}"
        print(f"❌ Error: {e}")

    return state
