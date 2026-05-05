"""State definition for multi-agent reasoning workflow.

Defines the shared state structure used across orchestrator, worker, and synthesizer agents.
"""

from typing import Any, Dict, List, Optional, TypedDict


class MultiAgentState(TypedDict, total=False):
    """State for the multi-agent knowledge graph reasoning workflow.

    This state is shared across all agents in the workflow and contains:
    - User input and configuration
    - Orchestrator outputs (entities, predicates, strategies)
    - Worker outputs (query results, markdown files)
    - Synthesizer outputs (final answer)
    - Status and error tracking
    """

    # ===== User Input =====
    user_query: str  # Original user question
    llm_provider: str  # LLM provider (e.g., "openai")
    llm_model_orchestrator: str  # Model for orchestrator
    llm_model_worker: str  # Model for workers
    llm_model_synthesizer: str  # Model for synthesizer

    # ===== Qdrant Configuration =====
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]

    # ===== Neo4j Configuration =====
    neo4j_uri: Optional[str]
    neo4j_user: Optional[str]
    neo4j_password: Optional[str]

    # ===== Orchestrator Outputs =====
    entities_found: List[str]  # List of canonical entity IDs
    predicates_found: List[str]  # List of canonical predicate IDs
    communities_identified: List[str]  # List of community IDs
    strategies: List[Dict[str, Any]]  # List of query strategies (max 5)
    orchestrator_raw_output: str  # Raw output from orchestrator
    resolution_method: str  # How entities/predicates were resolved: "matched", "partial_match", "fallback_top_connected"

    # ===== Worker Outputs =====
    worker_results: List[Dict[str, Any]]  # Results from each worker
    markdown_files: List[str]  # Paths to created markdown files
    total_results_count: int  # Total number of query results across all workers
    failed_strategies: List[str]  # Names of strategies that failed

    # ===== Synthesizer Outputs =====
    final_answer: str  # Synthesized final answer
    synthesis_quality: str  # Quality assessment (high/medium/basic/poor)
    files_read: int  # Number of markdown files read
    results_analyzed: int  # Number of individual results analyzed

    # ===== Status Tracking =====
    current_step: str  # Current workflow step
    status: str  # Overall status (running/success/error)
    error: Optional[str]  # Error message if status is error
    started_at: str  # ISO timestamp when workflow started
    completed_at: Optional[str]  # ISO timestamp when workflow completed

    # ===== Metadata =====
    workflow_version: str  # Version of the workflow
    execution_time_seconds: Optional[float]  # Total execution time


def create_initial_state(
    user_query: str,
    llm_provider: str = "openai",
    llm_model_orchestrator: str = "gpt-4o",
    llm_model_worker: str = "gpt-4o-mini",
    llm_model_synthesizer: str = "gpt-4o",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> MultiAgentState:
    """Create initial state for the workflow.

    Args:
        user_query: The user's question
        llm_provider: LLM provider to use
        llm_model_orchestrator: Model for orchestrator agent
        llm_model_worker: Model for worker agents
        llm_model_synthesizer: Model for synthesizer agent
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        neo4j_uri: Neo4j server URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Initial state dictionary
    """
    from datetime import datetime

    return MultiAgentState(
        # User input
        user_query=user_query,
        llm_provider=llm_provider,
        llm_model_orchestrator=llm_model_orchestrator,
        llm_model_worker=llm_model_worker,
        llm_model_synthesizer=llm_model_synthesizer,
        # Configuration
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        # Initialize outputs
        entities_found=[],
        predicates_found=[],
        communities_identified=[],
        strategies=[],
        orchestrator_raw_output="",
        resolution_method="",
        worker_results=[],
        markdown_files=[],
        total_results_count=0,
        failed_strategies=[],
        final_answer="",
        synthesis_quality="",
        files_read=0,
        results_analyzed=0,
        # Status
        current_step="initializing",
        status="running",
        error=None,
        started_at=datetime.now().isoformat(),
        completed_at=None,
        # Metadata
        workflow_version="1.0.0",
        execution_time_seconds=None,
    )


def update_state_from_orchestrator(
    state: MultiAgentState,
    orchestrator_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with orchestrator results.

    Args:
        state: Current state
        orchestrator_output: Output from orchestrator agent

    Returns:
        Updated state
    """
    state["entities_found"] = orchestrator_output.get("entities_found", [])
    state["predicates_found"] = orchestrator_output.get("predicates_found", [])
    state["communities_identified"] = orchestrator_output.get("communities_identified", [])
    state["strategies"] = orchestrator_output.get("strategies", [])
    state["orchestrator_raw_output"] = orchestrator_output.get("raw_output", "")
    state["resolution_method"] = orchestrator_output.get("resolution_method", "unknown")
    state["current_step"] = "orchestrator_complete"

    return state


def update_state_from_worker(
    state: MultiAgentState,
    worker_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with a worker's results.

    Args:
        state: Current state
        worker_output: Output from a worker agent

    Returns:
        Updated state
    """
    # Append worker result
    if "worker_results" not in state:
        state["worker_results"] = []
    state["worker_results"].append(worker_output)

    # Track markdown files
    if worker_output.get("markdown_file"):
        if "markdown_files" not in state:
            state["markdown_files"] = []
        state["markdown_files"].append(worker_output["markdown_file"])

    # Track results count
    state["total_results_count"] = state.get("total_results_count", 0) + worker_output.get(
        "results_count", 0
    )

    # Track failures
    if worker_output.get("status") != "success":
        if "failed_strategies" not in state:
            state["failed_strategies"] = []
        state["failed_strategies"].append(worker_output.get("strategy_name", "unknown"))

    return state


def update_state_from_synthesizer(
    state: MultiAgentState,
    synthesizer_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with synthesizer results.

    Args:
        state: Current state
        synthesizer_output: Output from synthesizer agent

    Returns:
        Updated state
    """
    from datetime import datetime

    state["final_answer"] = synthesizer_output.get("answer", "")
    state["synthesis_quality"] = synthesizer_output.get("synthesis_quality", "unknown")
    state["files_read"] = synthesizer_output.get("files_read", 0)
    state["results_analyzed"] = synthesizer_output.get("results_analyzed", 0)
    state["current_step"] = "complete"
    state["status"] = "success"
    state["completed_at"] = datetime.now().isoformat()

    # Calculate execution time
    if state.get("started_at") and state.get("completed_at"):
        from datetime import datetime

        start = datetime.fromisoformat(state["started_at"])
        end = datetime.fromisoformat(state["completed_at"])
        state["execution_time_seconds"] = (end - start).total_seconds()

    return state
