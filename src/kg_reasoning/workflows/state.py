"""State definition for multi-agent reasoning workflow.

Defines the shared state structure used across preprocessor, orchestrator,
worker, aggregator, and synthesizer agents.

CHANGES FROM ORIGINAL:
- Added preprocessor_chunk_ids: surfaces evidence point IDs that were
  previously discarded after entity/predicate extraction. These are what
  let the orchestrator call fetch_chunks / surround_chunks instead of only
  ever hopping the graph.
- Added preprocessor_chunk_details: per-chunk metadata (score, source query,
  community_id) so downstream tools don't need to re-fetch it.
- Added preprocessor_chunk_texts: the actual S3 source chunk text for every
  filtered chunk_id, fetched directly inside the preprocessor (Step 9 of
  its pipeline) using each chunk's community_id + chunk_id. Keyed by the
  same chunk_id as preprocessor_chunk_details. This is what lets the
  aggregator attach real reference text to its grouped output.
- Renamed the orchestrator's output from `strategies` (named-approach
  strings the worker switched on) to `worker_specs` (structured tool calls
  the worker dispatches via TOOL_MAP). `strategies` is kept as a
  backward-compatible alias so existing code that reads `state["strategies"]`
  doesn't break during migration — see update_state_from_orchestrator.
- Added aggregated_filepath: the single ranked/deduplicated markdown file
  the aggregator produces, consumed directly by the simplified synthesizer.
"""

from typing import Any, Dict, List, Optional, TypedDict

from kg_extractor.utils.model_setup import (
    REASONING_PROVIDER,
    ORCHESTRATOR_MODEL,
    SYNTHESIZER_MODEL,
    PREPROCESSING_MODEL,
)


class MultiAgentState(TypedDict, total=False):
    """State for the multi-agent knowledge graph reasoning workflow."""

    # ===== User Input =====
    user_query: str
    llm_provider: str
    llm_model_orchestrator: str
    llm_model_synthesizer: str
    llm_model_preprocessor: str

    # ===== PreProcessor Outputs =====
    expanded_query: str
    preprocessor_entity_ids: List[str]
    preprocessor_predicate_ids: List[str]
    preprocessor_raw_entity_names: List[str]  # NEW: raw surface forms for Qdrant evidence filtering
    preprocessor_raw_predicate_names: List[str]  # NEW: raw surface forms for Qdrant evidence filtering
    preprocessor_community_ids: List[str]
    preprocessor_community_labels: Dict[str, Any]  # NEW: community_id -> human-readable label (e.g. village name), from metadata_registry
    preprocessor_entity_details: Dict[str, Any]
    preprocessor_predicate_details: Dict[str, Any]
    preprocessor_entities_by_community: Dict[str, Any]  # NEW: community_id -> [canonical entity_ids], for per-community orchestrator specs
    preprocessor_predicates_by_community: Dict[str, Any]  # NEW: community_id -> [canonical predicate_ids], same purpose
    preprocessor_chunk_ids: List[str]  # NEW: evidence point IDs surfaced for chunk-level tools
    preprocessor_chunk_details: Dict[str, Any]  # NEW: chunk_id -> {score, query, community_id}
    preprocessor_chunk_texts: Dict[str, Any]  # NEW: chunk_id -> {community_id, text, s3_key, error} fetched from S3
    preprocessor_query_type: str
    preprocessor_needs_pathing: bool
    preprocessor_needs_community: bool

    # ===== Qdrant Configuration =====
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]

    # ===== Neo4j Configuration =====
    neo4j_uri: Optional[str]
    neo4j_user: Optional[str]
    neo4j_password: Optional[str]

    # ===== Orchestrator Outputs =====
    entities_found: List[str]
    predicates_found: List[str]
    communities_identified: List[str]
    worker_specs: List[Dict[str, Any]]  # NEW: structured {tool, ...params} dicts
    strategies: List[Dict[str, Any]]  # DEPRECATED alias for worker_specs — kept for back-compat
    orchestrator_raw_output: str
    resolution_method: str

    # ===== Worker Outputs =====
    worker_results: List[Dict[str, Any]]
    markdown_files: List[str]
    total_results_count: int
    failed_strategies: List[str]

    # ===== Aggregator Outputs =====
    aggregated_filepath: Optional[str]  # NEW: single ranked/deduped MD file for synthesizer
    aggregated_result_count: int  # NEW: how many unique results survived dedup

    # ===== Synthesizer Outputs =====
    final_answer: str
    synthesis_quality: str
    files_read: int
    results_analyzed: int

    # ===== Status Tracking =====
    current_step: str
    status: str
    error: Optional[str]
    started_at: str
    completed_at: Optional[str]

    # ===== Metadata =====
    workflow_version: str
    execution_time_seconds: Optional[float]


def create_initial_state(
    user_query: str,
    llm_provider: str = REASONING_PROVIDER,
    llm_model_orchestrator: str = ORCHESTRATOR_MODEL,
    llm_model_synthesizer: str = SYNTHESIZER_MODEL,
    llm_model_preprocessor: str = PREPROCESSING_MODEL,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> MultiAgentState:
    """Create initial state for the workflow."""
    from datetime import datetime

    return MultiAgentState(
        # User input
        user_query=user_query,
        llm_provider=llm_provider,
        llm_model_orchestrator=llm_model_orchestrator,
        llm_model_synthesizer=llm_model_synthesizer,
        llm_model_preprocessor=llm_model_preprocessor,
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
        worker_specs=[],
        strategies=[],
        orchestrator_raw_output="",
        resolution_method="",
        worker_results=[],
        markdown_files=[],
        total_results_count=0,
        failed_strategies=[],
        aggregated_filepath=None,
        aggregated_result_count=0,
        final_answer="",
        synthesis_quality="",
        files_read=0,
        results_analyzed=0,
        # PreProcessor outputs
        expanded_query="",
        preprocessor_entity_ids=[],
        preprocessor_predicate_ids=[],
        preprocessor_raw_entity_names=[],
        preprocessor_raw_predicate_names=[],
        preprocessor_community_ids=[],
        preprocessor_community_labels={},
        preprocessor_entity_details={},
        preprocessor_predicate_details={},
        preprocessor_entities_by_community={},
        preprocessor_predicates_by_community={},
        preprocessor_chunk_ids=[],
        preprocessor_chunk_details={},
        preprocessor_chunk_texts={},
        preprocessor_query_type="general",
        preprocessor_needs_pathing=False,
        preprocessor_needs_community=False,
        # Status
        current_step="initializing",
        status="running",
        error=None,
        started_at=datetime.now().isoformat(),
        completed_at=None,
        # Metadata
        workflow_version="2.0.0",
        execution_time_seconds=None,
    )


def update_state_from_orchestrator(
    state: MultiAgentState,
    orchestrator_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with orchestrator results.

    Reads `worker_specs` from the orchestrator output. `strategies` is
    populated as an identical alias so any code still reading the old key
    name keeps working during migration.
    """
    state["entities_found"] = orchestrator_output.get("entities_found", [])
    state["predicates_found"] = orchestrator_output.get("predicates_found", [])
    state["communities_identified"] = orchestrator_output.get("communities_identified", [])

    specs = orchestrator_output.get("worker_specs", orchestrator_output.get("strategies", []))
    state["worker_specs"] = specs
    state["strategies"] = specs  # back-compat alias, same list object

    state["orchestrator_raw_output"] = orchestrator_output.get("raw_output", "")
    state["resolution_method"] = orchestrator_output.get("resolution_method", "unknown")
    state["current_step"] = "orchestrator_complete"

    return state


def update_state_from_preprocessor(
    state: MultiAgentState,
    preprocessor_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with pre-processor results, including chunk IDs."""
    state["expanded_query"] = preprocessor_output.get("expanded_query", "")
    state["preprocessor_entity_ids"] = preprocessor_output.get("entity_ids", [])
    state["preprocessor_predicate_ids"] = preprocessor_output.get("predicate_ids", [])
    state["preprocessor_raw_entity_names"] = preprocessor_output.get("raw_entity_names", [])
    state["preprocessor_raw_predicate_names"] = preprocessor_output.get("raw_predicate_names", [])
    state["preprocessor_community_ids"] = preprocessor_output.get("community_ids", [])
    state["preprocessor_community_labels"] = preprocessor_output.get("community_labels", {})
    state["preprocessor_entity_details"] = preprocessor_output.get("entity_details", {})
    state["preprocessor_predicate_details"] = preprocessor_output.get("predicate_details", {})
    state["preprocessor_entities_by_community"] = preprocessor_output.get("entities_by_community", {})
    state["preprocessor_predicates_by_community"] = preprocessor_output.get("predicates_by_community", {})
    state["preprocessor_chunk_ids"] = preprocessor_output.get("chunk_ids", [])
    state["preprocessor_chunk_details"] = preprocessor_output.get("chunk_details", {})
    state["preprocessor_chunk_texts"] = preprocessor_output.get("chunk_texts", {})
    state["preprocessor_query_type"] = preprocessor_output.get("query_type", "general")
    state["preprocessor_needs_pathing"] = preprocessor_output.get("needs_pathing", False)
    state["preprocessor_needs_community"] = preprocessor_output.get("needs_community", False)
    state["current_step"] = "preprocessor_complete"

    return state


def update_state_from_worker(
    state: MultiAgentState,
    worker_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with a worker's results."""
    if "worker_results" not in state:
        state["worker_results"] = []
    state["worker_results"].append(worker_output)

    if worker_output.get("markdown_file"):
        if "markdown_files" not in state:
            state["markdown_files"] = []
        state["markdown_files"].append(worker_output["markdown_file"])

    state["total_results_count"] = state.get("total_results_count", 0) + worker_output.get(
        "results_count", 0
    )

    if worker_output.get("status") != "success":
        if "failed_strategies" not in state:
            state["failed_strategies"] = []
        state["failed_strategies"].append(worker_output.get("strategy_name", "unknown"))

    return state


def update_state_from_aggregator(
    state: MultiAgentState,
    aggregator_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with the aggregator's single ranked/deduplicated file."""
    state["aggregated_filepath"] = aggregator_output.get("filepath")
    state["aggregated_result_count"] = aggregator_output.get("result_count", 0)
    state["current_step"] = "aggregator_complete"
    return state


def update_state_from_synthesizer(
    state: MultiAgentState,
    synthesizer_output: Dict[str, Any],
) -> MultiAgentState:
    """Update state with synthesizer results."""
    from datetime import datetime

    state["final_answer"] = synthesizer_output.get("answer", "")
    state["synthesis_quality"] = synthesizer_output.get("synthesis_quality", "unknown")
    state["files_read"] = synthesizer_output.get("files_read", 0)
    state["results_analyzed"] = synthesizer_output.get("results_analyzed", 0)
    state["current_step"] = "complete"
    state["status"] = "success"
    state["completed_at"] = datetime.now().isoformat()

    if state.get("started_at") and state.get("completed_at"):
        start = datetime.fromisoformat(state["started_at"])
        end = datetime.fromisoformat(state["completed_at"])
        state["execution_time_seconds"] = (end - start).total_seconds()

    return state