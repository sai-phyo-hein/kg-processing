"""Triple extraction node for LangGraph workflow."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.extract_triples import extract_triples_from_chunks

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def extract_triples_node(state: "WorkflowState") -> "WorkflowState":
    """Extract knowledge graph triples from chunks."""
    try:
        print("🔗 Extracting triples...")

        if not state["chunks_path"]:
            raise ValueError("No chunks path available")

        path = Path(state["chunks_path"])
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {state['chunks_path']}")

        # Read chunks
        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = chunks_data.get("chunks", [])
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
