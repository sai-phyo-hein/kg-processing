"""Triple translation node for LangGraph workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.translate_triples import translate_triples_from_file

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def translate_triples_node(state: "WorkflowState") -> "WorkflowState":
    """Translate triple fields to English in place.

    Reads the triples file written by the extraction node, translates the
    chunks that contain non-English text (detected per chunk with lingua), and
    writes the result back to the same ``triples_path`` (translation only adds
    ``_en`` fields).

    When ``state["chunk_id"]`` is set, only that chunk is (re-)translated in
    place; every other chunk is preserved.
    """
    try:
        print("🌐 Translating triples to English...")

        if not state["triples_path"]:
            raise ValueError("No triples path available")

        path = Path(state["triples_path"])
        if not path.exists():
            raise FileNotFoundError(f"Triples file not found: {state['triples_path']}")

        # Optional single-chunk (re-)translation: only this chunk is translated
        # and written back in place (all other chunks preserved).
        chunk_id = state.get("chunk_id")
        if chunk_id is not None:
            print(f"🎯 Translating only chunk_id {chunk_id} (other chunks preserved)")

        # In-place: translates non-English chunks and writes back to the same path.
        output_path = translate_triples_from_file(
            input_path=state["triples_path"],
            chunk_id=chunk_id,
        )

        # triples_path is unchanged (in-place). Translation always precedes
        # refinement, so the next step is always refine_triples.
        state["current_step"] = "refine_triples"
        print(f"✅ Triple translation completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Triple translation failed: {e}"
        print(f"❌ Error: {e}")

    return state
