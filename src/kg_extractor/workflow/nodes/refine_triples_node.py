"""Triple refinement node for LangGraph workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.refine_triples import refine_triples_from_file

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def refine_triples_node(state: "WorkflowState") -> "WorkflowState":
    """Refine triples using Qdrant for entity resolution."""
    try:
        print("🔍 Refining triples with Qdrant...")

        if not state["triples_path"]:
            raise ValueError("No triples path available")

        path = Path(state["triples_path"])
        if not path.exists():
            raise FileNotFoundError(f"Triples file not found: {state['triples_path']}")

        # Refine triples
        output_path = refine_triples_from_file(
            input_path=state["triples_path"],
            llm_provider=state["refinement_llm_provider"],
            llm_model=state["refinement_llm_model"],
            similarity_threshold=state["similarity_threshold"],
        )

        state["refined_path"] = output_path
        state["current_step"] = "build_graph"
        print(f"✅ Triple refinement completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Triple refinement failed: {e}"
        print(f"❌ Error: {e}")

    return state
