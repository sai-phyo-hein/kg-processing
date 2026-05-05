"""Graph building node for LangGraph workflow."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.build_graph import build_graph_from_file

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def build_graph_node(state: "WorkflowState") -> "WorkflowState":
    """Build Neo4j graph from refined triples."""
    try:
        print("🕸️ Building Neo4j graph...")

        if not state["refined_path"]:
            raise ValueError("No refined triples path available")

        path = Path(state["refined_path"])
        if not path.exists():
            raise FileNotFoundError(f"Refined triples file not found: {state['refined_path']}")

        # Get Neo4j connection parameters from environment
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not neo4j_uri or not neo4j_user or not neo4j_password:
            raise ValueError("NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME), and NEO4J_PASSWORD must be set in environment variables")

        # Build graph
        stats = build_graph_from_file(
            state["refined_path"],
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            with_schema=state["with_schema"],
        )

        state["graph_stats"] = stats
        state["current_step"] = "complete"
        state["status"] = "success"
        print(f"✅ Graph building completed")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Graph building failed: {e}"
        print(f"❌ Error: {e}")

    return state
