"""LangGraph-based workflow for document processing and knowledge graph extraction."""

from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from kg_extractor.utils.model_setup import (
    PARSING_PROVIDER,
    PARSING_MODEL,
    CHUNKING_PROVIDER,
    CHUNKING_MODEL,
    TRIPLET_PROVIDER,
    TRIPLET_MODEL,
    REFINEMENT_PROVIDER,
    REFINEMENT_MODEL,
)
from .nodes import (
    process_document_node,
    extract_metadata_node,
    chunk_document_node,
    extract_triples_node,
    refine_triples_node,
    build_graph_node,
)

# Load environment variables
load_dotenv()


class WorkflowState(TypedDict):
    """State for the document processing workflow."""

    input_file: str
    provider: str
    model: str
    chunk_granularity: float
    similarity_threshold: float
    chunking_llm_provider: str
    chunking_llm_model: str
    triplet_llm_provider: str
    triplet_llm_model: str
    refine_triples: bool
    refinement_llm_provider: str
    refinement_llm_model: str
    build_graph: bool
    output_dir: str
    until_step: Literal["document_parsing", "metadata_extraction", "semantic_chunking", "triple_extraction", "triple_refining", "graph_building"] | None
    pages: List[int] | None
    location_moo: str | None
    location_village: str | None

    # Intermediate results
    markdown_path: str | None
    metadata: Dict[str, Any] | None
    chunks_path: str | None
    triples_path: str | None
    refined_path: str | None
    graph_stats: Dict[str, Any] | None

    # Status
    status: str
    error: str | None
    current_step: str


def should_continue_to_next_step(state: WorkflowState, current_step: str) -> str:
    """Determine whether to continue to the next step or stop at until_step."""
    if state["status"] == "error":
        return "error"

    if state["until_step"] is None:
        return "continue"

    step_mapping = {
        "process_document": "document_parsing",
        "extract_metadata": "metadata_extraction",
        "chunk_document": "semantic_chunking",
        "extract_triples": "triple_extraction",
        "refine_triples": "triple_refining",
        "build_graph": "graph_building",
    }

    current_step_name = step_mapping.get(current_step)

    if current_step_name == state["until_step"]:
        return "complete"

    return "continue"


def create_langgraph_workflow() -> StateGraph:
    """Create a LangGraph workflow for document processing."""

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("process_document", process_document_node)
    workflow.add_node("extract_metadata", extract_metadata_node)
    workflow.add_node("chunk_document", chunk_document_node)
    workflow.add_node("extract_triples", extract_triples_node)
    workflow.add_node("refine_triples", refine_triples_node)
    workflow.add_node("build_graph", build_graph_node)

    # Define the edges
    workflow.set_entry_point("extract_metadata")

    workflow.add_conditional_edges(
        "extract_metadata",
        lambda state: should_continue_to_next_step(state, "extract_metadata"),
        {"continue": "process_document", "complete": END, "error": END},
    )

    workflow.add_conditional_edges(
        "process_document",
        lambda state: should_continue_to_next_step(state, "process_document"),
        {"continue": "chunk_document", "complete": END, "error": END},
    )

    workflow.add_conditional_edges(
        "chunk_document",
        lambda state: should_continue_to_next_step(state, "chunk_document"),
        {"continue": "extract_triples", "complete": END, "error": END},
    )

    workflow.add_conditional_edges(
        "extract_triples",
        lambda state: should_continue_to_next_step(state, "extract_triples"),
        {"continue": "refine_triples", "complete": END, "error": END},
    )

    workflow.add_conditional_edges(
        "refine_triples",
        lambda state: should_continue_to_next_step(state, "refine_triples"),
        {"continue": "build_graph", "complete": END, "error": END},
    )

    workflow.add_conditional_edges(
        "build_graph",
        lambda state: should_continue_to_next_step(state, "build_graph") if state["status"] != "error" else "error",
        {"continue": END, "complete": END, "error": END},
    )

    return workflow


def run_langgraph_workflow(
    input_file: str,
    provider: str = PARSING_PROVIDER,
    model: str = PARSING_MODEL,
    chunk_granularity: float = 0.1,
    similarity_threshold: float = 0.95,
    chunking_llm_provider: str = CHUNKING_PROVIDER,
    chunking_llm_model: str = CHUNKING_MODEL,
    triplet_llm_provider: str = TRIPLET_PROVIDER,
    triplet_llm_model: str = TRIPLET_MODEL,
    refinement_llm_provider: str = REFINEMENT_PROVIDER,
    refinement_llm_model: str = REFINEMENT_MODEL,
    until_step: Literal["document_parsing", "metadata_extraction", "semantic_chunking", "triple_extraction", "triple_refining", "graph_building"] | None = None,
    pages: List[int] | None = None,
    location_moo: str | None = None,
    location_village: str | None = None,
) -> Dict[str, Any]:
    """Run the full LangGraph workflow for document processing.

    Runs all steps: parse → metadata → chunk → extract triples → refine triples → build graph.
    Use until_step to stop early. For single-node debug runs use run_refine_triples_only()
    or run_build_graph_only() instead.

    Args:
        input_file: Path to input document
        provider: API provider for document processing
        model: Model to use for document processing
        chunk_granularity: Granularity for semantic chunking (0.0=very fine, 1.0=coarse)
        similarity_threshold: Cosine similarity threshold for entity resolution in triple refiner
        chunking_llm_provider: LLM provider for chunking and metadata extraction
        chunking_llm_model: Model to use for chunking analysis and metadata extraction
        triplet_llm_provider: LLM provider for triple extraction
        triplet_llm_model: Model to use for triple extraction
        refinement_llm_provider: LLM provider for triple refinement
        refinement_llm_model: Model for triple refinement
        until_step: Stop workflow after this step (None to run all steps)
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Dictionary with processing results
    """
    # Create the workflow
    workflow = create_langgraph_workflow()
    app = workflow.compile()

    # Initialize state
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    initial_state: WorkflowState = {
        "input_file": input_file,
        "provider": provider,
        "model": model,
        "chunk_granularity": chunk_granularity,
        "similarity_threshold": similarity_threshold,
        "chunking_llm_provider": chunking_llm_provider,
        "chunking_llm_model": chunking_llm_model,
        "triplet_llm_provider": triplet_llm_provider,
        "triplet_llm_model": triplet_llm_model,
        "refine_triples": True,
        "refinement_llm_provider": refinement_llm_provider,
        "refinement_llm_model": refinement_llm_model,
        "build_graph": True,
        "output_dir": output_dir,
        "until_step": until_step,
        "pages": pages,
        "location_moo": location_moo,
        "location_village": location_village,
        "markdown_path": None,
        "metadata": None,
        "chunks_path": None,
        "triples_path": None,
        "refined_path": None,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "process_document",
    }

    # Run the workflow
    print("🚀 Starting LangGraph workflow...")
    final_state = app.invoke(initial_state)

    # Return results
    return {
        "markdown_output": final_state["markdown_path"],
        "metadata": final_state["metadata"],
        "chunks_output": final_state["chunks_path"],
        "triples_output": final_state["triples_path"],
        "refined_output": final_state["refined_path"],
        "graph_stats": final_state["graph_stats"],
        "status": final_state["status"],
        "error": final_state["error"],
    }


def run_parse_document_only(
    input_file: str,
    provider: str = PARSING_PROVIDER,
    model: str = PARSING_MODEL,
    pages: List[int] | None = None,
) -> Dict[str, Any]:
    """Debug mode: run only the parse-document node.

    Args:
        input_file: Path to the input document.
        provider: API provider for document parsing.
        model: Model to use for document parsing.
        pages: Optional list of page numbers to process (1-indexed).

    Returns:
        Dictionary with markdown_output, status, and error keys.
    """
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    state: WorkflowState = {
        "input_file": input_file,
        "provider": provider,
        "model": model,
        "chunk_granularity": 0.0,
        "similarity_threshold": 0.0,
        "chunking_llm_provider": "",
        "chunking_llm_model": "",
        "triplet_llm_provider": "",
        "triplet_llm_model": "",
        "refine_triples": False,
        "refinement_llm_provider": "",
        "refinement_llm_model": "",
        "build_graph": False,
        "output_dir": output_dir,
        "until_step": None,
        "pages": pages,
        "markdown_path": None,
        "metadata": None,
        "chunks_path": None,
        "triples_path": None,
        "refined_path": None,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "process_document",
    }

    final_state = process_document_node(state)

    if final_state["status"] != "error":
        final_state["status"] = "success"

    return {
        "markdown_output": final_state["markdown_path"],
        "status": final_state["status"],
        "error": final_state["error"],
    }


def run_extract_metadata_only(
    input_file: str,
    location_moo: str | None = None,
    location_village: str | None = None,
) -> Dict[str, Any]:
    """Debug mode: run only the extract-metadata node.

    Resolves the markdown file from a previous parse run using the naming convention
    ``output/{stem}_analysis.md``, runs the extract-metadata node, then stops.

    Args:
        input_file: Original document path (used to locate the existing markdown file).
        location_moo: Optional location_moo value from CLI to use instead of extraction.
        location_village: Optional location_village value from CLI to use instead of extraction.

    Returns:
        Dictionary with metadata, status, and error keys.
    """
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    stem = Path(input_file).stem
    markdown_path = str(Path(output_dir) / f"{stem}_analysis.md")

    state: WorkflowState = {
        "input_file": input_file,
        "provider": "",
        "model": "",
        "chunk_granularity": 0.0,
        "similarity_threshold": 0.0,
        "chunking_llm_provider": chunking_llm_provider,
        "chunking_llm_model": chunking_llm_model,
        "triplet_llm_provider": "",
        "triplet_llm_model": "",
        "refine_triples": False,
        "refinement_llm_provider": "",
        "refinement_llm_model": "",
        "build_graph": False,
        "output_dir": output_dir,
        "until_step": None,
        "pages": None,
        "location_moo": location_moo,
        "location_village": location_village,
        "markdown_path": markdown_path,
        "metadata": None,
        "chunks_path": None,
        "triples_path": None,
        "refined_path": None,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "extract_metadata",
    }

    final_state = extract_metadata_node(state)

    if final_state["status"] != "error":
        final_state["status"] = "success"

    return {
        "metadata": final_state["metadata"],
        "status": final_state["status"],
        "error": final_state["error"],
    }


def run_chunk_document_only(
    input_file: str,
    chunk_granularity: float = 0.1,
    chunking_llm_provider: str = CHUNKING_PROVIDER,
    chunking_llm_model: str = CHUNKING_MODEL,
) -> Dict[str, Any]:
    """Debug mode: run only the chunk-document node.

    Resolves the markdown file from a previous parse run using the naming convention
    ``output/{stem}_analysis.md``, runs the chunk-document node, then stops.

    Args:
        input_file: Original document path (used to locate the existing markdown file).
        chunk_granularity: Granularity for semantic chunking (0.0=very fine, 1.0=coarse).
        chunking_llm_provider: LLM provider for chunking.
        chunking_llm_model: Model for chunking.

    Returns:
        Dictionary with chunks_output, status, and error keys.
    """
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    stem = Path(input_file).stem
    markdown_path = str(Path(output_dir) / f"{stem}_analysis.md")

    state: WorkflowState = {
        "input_file": input_file,
        "provider": "",
        "model": "",
        "chunk_granularity": chunk_granularity,
        "similarity_threshold": 0.0,
        "chunking_llm_provider": chunking_llm_provider,
        "chunking_llm_model": chunking_llm_model,
        "triplet_llm_provider": "",
        "triplet_llm_model": "",
        "refine_triples": False,
        "refinement_llm_provider": "",
        "refinement_llm_model": "",
        "build_graph": False,
        "output_dir": output_dir,
        "until_step": None,
        "pages": None,
        "markdown_path": markdown_path,
        "metadata": None,
        "chunks_path": None,
        "triples_path": None,
        "refined_path": None,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "chunk_document",
    }

    final_state = chunk_document_node(state)

    if final_state["status"] != "error":
        final_state["status"] = "success"

    return {
        "chunks_output": final_state["chunks_path"],
        "status": final_state["status"],
        "error": final_state["error"],
    }


def run_extract_triples_only(
    input_file: str,
    triplet_llm_provider: str = TRIPLET_PROVIDER,
    triplet_llm_model: str = TRIPLET_MODEL,
) -> Dict[str, Any]:
    """Debug mode: run only the extract-triples node.

    Resolves the chunks file from a previous chunk run using the naming convention
    ``output/{stem}_chunks.json``, runs the extract-triples node, then stops.

    Args:
        input_file: Original document path (used to locate the existing chunks file).
        triplet_llm_provider: LLM provider for triple extraction.
        triplet_llm_model: Model for triple extraction.

    Returns:
        Dictionary with triples_output, status, and error keys.
    """
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    stem = Path(input_file).stem
    chunks_path = str(Path(output_dir) / f"{stem}_chunks")

    state: WorkflowState = {
        "input_file": input_file,
        "provider": "",
        "model": "",
        "chunk_granularity": 0.0,
        "similarity_threshold": 0.0,
        "chunking_llm_provider": "",
        "chunking_llm_model": "",
        "triplet_llm_provider": triplet_llm_provider,
        "triplet_llm_model": triplet_llm_model,
        "refine_triples": False,
        "refinement_llm_provider": "",
        "refinement_llm_model": "",
        "build_graph": False,
        "output_dir": output_dir,
        "until_step": None,
        "pages": None,
        "markdown_path": None,
        "metadata": None,
        "chunks_path": chunks_path,
        "triples_path": None,
        "refined_path": None,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "extract_triples",
    }

    final_state = extract_triples_node(state)

    if final_state["status"] != "error":
        final_state["status"] = "success"

    return {
        "triples_output": final_state["triples_path"],
        "status": final_state["status"],
        "error": final_state["error"],
    }


def run_refine_triples_only(
    input_file: str,
    refinement_llm_provider: str = REFINEMENT_PROVIDER,
    refinement_llm_model: str = REFINEMENT_MODEL,
    similarity_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Debug mode: run only the refine-triples node.

    Resolves the triples file from a previous pipeline run using the naming convention
    ``output/{stem}_triples.json``, runs the refine-triples node, then stops.

    Args:
        input_file: Original document path (used to locate the existing triples file).
        refinement_llm_provider: LLM provider for triple refinement.
        refinement_llm_model: Model for triple refinement.
        similarity_threshold: Cosine similarity threshold for entity resolution.

    Returns:
        Dictionary with refined_output, status, and error keys.
    """
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    stem = Path(input_file).stem
    triples_path = str(Path(output_dir) / f"{stem}_triples.json")

    state: WorkflowState = {
        "input_file": input_file,
        "provider": "",
        "model": "",
        "chunk_granularity": 0.0,
        "similarity_threshold": similarity_threshold,
        "chunking_llm_provider": "",
        "chunking_llm_model": "",
        "triplet_llm_provider": "",
        "triplet_llm_model": "",
        "refine_triples": True,
        "refinement_llm_provider": refinement_llm_provider,
        "refinement_llm_model": refinement_llm_model,
        "build_graph": False,
        "output_dir": output_dir,
        "until_step": None,
        "pages": None,
        "markdown_path": None,
        "metadata": None,
        "chunks_path": None,
        "triples_path": triples_path,
        "refined_path": None,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "refine_triples",
    }

    final_state = refine_triples_node(state)

    if final_state["status"] != "error":
        final_state["status"] = "success"

    return {
        "refined_output": final_state["refined_path"],
        "status": final_state["status"],
        "error": final_state["error"],
    }


def run_build_graph_only(
    input_file: str,
) -> Dict[str, Any]:
    """Debug mode: run only the build-graph node.

    Resolves the refined triples file from a previous pipeline run using the naming
    convention ``output/{stem}_triples_refined.json``, runs the build-graph node,
    then stops.

    Args:
        input_file: Original document path (used to locate the existing refined triples file).

    Returns:
        Dictionary with graph_stats, status, and error keys.
    """
    output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    stem = Path(input_file).stem
    refined_path = str(Path(output_dir) / f"{stem}_triples_refined.json")

    state: WorkflowState = {
        "input_file": input_file,
        "provider": "",
        "model": "",
        "chunk_granularity": 0.0,
        "similarity_threshold": 0.0,
        "chunking_llm_provider": "",
        "chunking_llm_model": "",
        "triplet_llm_provider": "",
        "triplet_llm_model": "",
        "refine_triples": False,
        "refinement_llm_provider": "",
        "refinement_llm_model": "",
        "build_graph": True,
        "output_dir": output_dir,
        "until_step": None,
        "pages": None,
        "markdown_path": None,
        "metadata": None,
        "chunks_path": None,
        "triples_path": None,
        "refined_path": refined_path,
        "graph_stats": None,
        "status": "in_progress",
        "error": None,
        "current_step": "build_graph",
    }

    final_state = build_graph_node(state)

    if final_state["status"] != "error":
        final_state["status"] = "success"

    return {
        "graph_stats": final_state["graph_stats"],
        "status": final_state["status"],
        "error": final_state["error"],
    }
