"""LangGraph-based workflow for document processing and knowledge graph extraction."""

import json
from pathlib import Path
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from kg_extractor.utils.input_processor import DocumentProcessor
from kg_extractor.utils.markdown_formatter import save_markdown_result, save_text_markdown
from kg_extractor.utils.parser import (
    NVIDIAAPIError,
    NVIDIAConfig,
    OpenRouterAPIError,
    OpenRouterConfig,
    OpenAIAPIError,
    OpenAIConfig,
    GoogleAPIError,
    GoogleConfig,
    extract_text_from_document,
    extract_text_from_document_openrouter,
    extract_text_from_document_openai,
    extract_text_from_document_google,
    get_api_key,
    get_openrouter_api_key,
    get_openai_api_key,
    get_google_api_key,
    process_document_with_api,
    process_document_with_openrouter,
    process_document_with_openai,
    process_document_with_google,
)
from kg_extractor.utils.semantic_chunker import chunk_markdown_file
from kg_extractor.utils.triple_extractor import extract_triples_from_chunks
from kg_extractor.utils.triple_refiner import refine_triples_from_file
from kg_extractor.utils.neo4j_graph_builder import build_graph_from_file

# Load environment variables
load_dotenv()


class WorkflowState(TypedDict):
    """State for the document processing workflow."""

    input_file: str
    provider: str
    model: str
    content_type: str
    similarity_threshold: float
    min_chunk_size: int
    max_chunk_size: int
    output_format: str
    chunking_llm_provider: str
    chunking_llm_model: str
    triplet_llm_provider: str
    triplet_llm_model: str
    refine_triples: bool
    refinement_llm_provider: str
    refinement_llm_model: str
    build_graph: bool
    with_schema: bool
    output_dir: str

    # Intermediate results
    markdown_path: str | None
    chunks_path: str | None
    triples_path: str | None
    refined_path: str | None
    graph_stats: Dict[str, Any] | None

    # Status
    status: str
    error: str | None
    current_step: str


def process_document_node(state: WorkflowState) -> WorkflowState:
    """Process document and extract structured content."""
    try:
        print("📄 Processing document...")

        path = Path(state["input_file"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {state['input_file']}")

        # Get file type
        file_type = DocumentProcessor.get_file_type(state["input_file"])
        if file_type == "unknown":
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Get API key and create configuration
        if state["provider"] == "nvidia":
            api_key = get_api_key()
            config = NVIDIAConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        elif state["provider"] == "openrouter":
            api_key = get_openrouter_api_key()
            config = OpenRouterConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        elif state["provider"] == "openai":
            api_key = get_openai_api_key()
            config = OpenAIConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        else:  # google
            api_key = get_google_api_key()
            config = GoogleConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )

        # Process document based on output format
        if state["output_format"] == "json":
            if state["provider"] == "nvidia":
                result = extract_text_from_document(
                    state["input_file"], config, state["content_type"]
                )
            elif state["provider"] == "openrouter":
                result = extract_text_from_document_openrouter(
                    state["input_file"], config, state["content_type"]
                )
            elif state["provider"] == "openai":
                result = extract_text_from_document_openai(
                    state["input_file"], config, state["content_type"]
                )
            else:  # google
                result = extract_text_from_document_google(
                    state["input_file"], config, state["content_type"]
                )

            # Save JSON result
            output_file = Path(state["output_dir"]) / f"{path.stem}_analysis.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            state["markdown_path"] = str(output_file)

        elif state["output_format"] == "markdown":
            if state["provider"] == "nvidia":
                result = process_document_with_api(
                    state["input_file"], config, state["content_type"]
                )
            elif state["provider"] == "openrouter":
                result = process_document_with_openrouter(
                    state["input_file"], config, state["content_type"]
                )
            elif state["provider"] == "openai":
                result = process_document_with_openai(
                    state["input_file"], config, state["content_type"]
                )
            else:  # google
                result = process_document_with_google(
                    state["input_file"], config, state["content_type"]
                )

            # Save markdown result
            output_file = save_markdown_result(result, state["input_file"])
            state["markdown_path"] = output_file

        else:  # text format
            if state["provider"] == "nvidia":
                text = extract_text_from_document(
                    state["input_file"], config, state["content_type"]
                )
            elif state["provider"] == "openrouter":
                text = extract_text_from_document_openrouter(
                    state["input_file"], config, state["content_type"]
                )
            elif state["provider"] == "openai":
                text = extract_text_from_document_openai(
                    state["input_file"], config, state["content_type"]
                )
            else:  # google
                text = extract_text_from_document_google(
                    state["input_file"], config, state["content_type"]
                )

            # Save text result
            output_file = save_text_markdown(text, state["input_file"])
            state["markdown_path"] = output_file

        state["current_step"] = "chunk_document"
        print(f"✅ Document processed: {state['markdown_path']}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Document processing failed: {e}"
        print(f"❌ Error: {e}")

    return state


def chunk_document_node(state: WorkflowState) -> WorkflowState:
    """Perform semantic chunking on the processed document."""
    try:
        print("🧩 Performing semantic chunking...")

        if not state["markdown_path"]:
            raise ValueError("No markdown path available")

        path = Path(state["markdown_path"])
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {state['markdown_path']}")

        # Perform chunking
        output_path = chunk_markdown_file(
            file_path=state["markdown_path"],
            similarity_threshold=state["similarity_threshold"],
            min_chunk_size=state["min_chunk_size"],
            max_chunk_size=state["max_chunk_size"],
            llm_provider=state["chunking_llm_provider"],
            llm_model=state["chunking_llm_model"],
            output_dir=state["output_dir"],
        )

        state["chunks_path"] = output_path
        state["current_step"] = "extract_triples"
        print(f"✅ Chunking completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Chunking failed: {e}"
        print(f"❌ Error: {e}")

    return state


def extract_triples_node(state: WorkflowState) -> WorkflowState:
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

        # Extract triples
        output_path = extract_triples_from_chunks(
            chunks=chunks,
            source_file=state["input_file"],
            llm_provider=state["triplet_llm_provider"],
            llm_model=state["triplet_llm_model"],
            output_dir=state["output_dir"],
            with_schema=state["with_schema"],
        )

        state["triples_path"] = output_path
        state["current_step"] = "refine_triples" if state["refine_triples"] else "build_graph"
        print(f"✅ Triple extraction completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Triple extraction failed: {e}"
        print(f"❌ Error: {e}")

    return state


def refine_triples_node(state: WorkflowState) -> WorkflowState:
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
        )

        state["refined_path"] = output_path
        state["current_step"] = "build_graph"
        print(f"✅ Triple refinement completed: {output_path}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Triple refinement failed: {e}"
        print(f"❌ Error: {e}")

    return state


def build_graph_node(state: WorkflowState) -> WorkflowState:
    """Build Neo4j graph from refined triples."""
    try:
        print("🕸️ Building Neo4j graph...")

        if not state["refined_path"]:
            raise ValueError("No refined triples path available")

        path = Path(state["refined_path"])
        if not path.exists():
            raise FileNotFoundError(f"Refined triples file not found: {state['refined_path']}")

        # Build graph
        stats = build_graph_from_file(state["refined_path"], with_schema=state["with_schema"])

        state["graph_stats"] = stats
        state["current_step"] = "complete"
        state["status"] = "success"
        print(f"✅ Graph building completed")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Graph building failed: {e}"
        print(f"❌ Error: {e}")

    return state


def should_refine_triples(state: WorkflowState) -> str:
    """Determine whether to proceed to triple refinement."""
    if state["status"] == "error":
        return "error"

    if state["refine_triples"]:
        return "refine_triples"
    else:
        return "skip_refine"


def should_build_graph(state: WorkflowState) -> str:
    """Determine whether to proceed to graph building."""
    if state["status"] == "error":
        return "error"

    if state["build_graph"] and state["refined_path"]:
        return "build_graph"
    else:
        return "complete"


def create_langgraph_workflow() -> StateGraph:
    """Create a LangGraph workflow for document processing."""

    # Create the workflow graph
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("process_document", process_document_node)
    workflow.add_node("chunk_document", chunk_document_node)
    workflow.add_node("extract_triples", extract_triples_node)
    workflow.add_node("refine_triples", refine_triples_node)
    workflow.add_node("build_graph", build_graph_node)

    # Define the edges
    workflow.set_entry_point("process_document")

    workflow.add_edge("process_document", "chunk_document")
    workflow.add_edge("chunk_document", "extract_triples")

    # Conditional edge for triple refinement
    workflow.add_conditional_edges(
        "extract_triples",
        should_refine_triples,
        {
            "refine_triples": "refine_triples",
            "skip_refine": "build_graph",
            "error": END,
        },
    )

    # Conditional edge for graph building
    workflow.add_conditional_edges(
        "refine_triples",
        should_build_graph,
        {
            "build_graph": "build_graph",
            "complete": END,
            "error": END,
        },
    )

    workflow.add_conditional_edges(
        "build_graph",
        should_build_graph,
        {
            "complete": END,
            "error": END,
        },
    )

    return workflow


def run_langgraph_workflow(
    input_file: str,
    provider: str = "nvidia",
    model: str = "microsoft/phi-4-multimodal-instruct",
    content_type: str = "mixed",
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
    output_format: str = "markdown",
    chunking_llm_provider: str = "openai",
    chunking_llm_model: str = "gpt-4o-mini",
    triplet_llm_provider: str = "openai",
    triplet_llm_model: str = "gpt-4o-mini",
    refine_triples: bool = True,
    refinement_llm_provider: str = "openai",
    refinement_llm_model: str = "gpt-4o-mini",
    build_graph: bool = True,
    with_schema: bool = False,
) -> Dict[str, Any]:
    """Run the LangGraph workflow for document processing.

    Args:
        input_file: Path to input document
        provider: API provider for document processing
        model: Model to use for document processing
        content_type: Type of content to extract
        similarity_threshold: Threshold for topic change detection
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
        output_format: Output format for document processing
        chunking_llm_provider: LLM provider for chunking
        chunking_llm_model: Model to use for chunking analysis
        triplet_llm_provider: LLM provider for triple extraction
        triplet_llm_model: Model to use for triple extraction
        refine_triples: Whether to refine triples using Qdrant
        refinement_llm_provider: LLM provider for triple refinement
        refinement_llm_model: Model for triple refinement
        build_graph: Whether to build Neo4j graph from refined triples
        with_schema: Whether to validate triples and graph against schema.md

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
        "content_type": content_type,
        "similarity_threshold": similarity_threshold,
        "min_chunk_size": min_chunk_size,
        "max_chunk_size": max_chunk_size,
        "output_format": output_format,
        "chunking_llm_provider": chunking_llm_provider,
        "chunking_llm_model": chunking_llm_model,
        "triplet_llm_provider": triplet_llm_provider,
        "triplet_llm_model": triplet_llm_model,
        "refine_triples": refine_triples,
        "refinement_llm_provider": refinement_llm_provider,
        "refinement_llm_model": refinement_llm_model,
        "build_graph": build_graph,
        "with_schema": with_schema,
        "output_dir": output_dir,
        "markdown_path": None,
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
        "chunks_output": final_state["chunks_path"],
        "triples_output": final_state["triples_path"],
        "refined_output": final_state["refined_path"],
        "graph_stats": final_state["graph_stats"],
        "status": final_state["status"],
        "error": final_state["error"],
    }
