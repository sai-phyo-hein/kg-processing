"""Multi-agent workflow using LangGraph.

Orchestrates the flow:
  Preprocessor → Orchestrator → Workers (parallel) → Aggregator → Synthesizer

CHANGES FROM ORIGINAL:
- Added aggregator_node between workers and synthesizer. Workers write N
  individual markdown files; the aggregator deduplicates and ranks them into
  one clean file; the synthesizer reads exactly that one file. This removes
  the synthesizer's need to decide which files to read (it was doing so via
  a tool call inside a ReAct loop — now it's a plain Python pass with no
  LLM involved).
- Workers now dispatch on state["worker_specs"] (new WorkerSpec shape) but
  also accept state["strategies"] (legacy shape) as a fallback, since the
  ReAct-based orchestrator path still emits the latter. WorkerAgent._normalize
  handles the translation so the execution path is identical either way.
- Synthesizer now receives aggregated_filepath from state and makes a single
  non-agentic LLM call. create_react_agent wrapper is gone.
- update_state_from_aggregator is threaded into the workflow state.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from kg_extractor.utils.model_setup import (
    REASONING_PROVIDER,
    ORCHESTRATOR_MODEL,
    SYNTHESIZER_MODEL,
    PREPROCESSING_MODEL,
)
from kg_reasoning.agents.orchestrator import OrchestratorAgent
from kg_reasoning.agents.preprocessor import PreProcessor
from kg_reasoning.agents.worker import WorkerAgent, execute_strategy_parallel
from kg_reasoning.agents.synthesizer import SynthesizerAgent
from kg_reasoning.agents.aggregator import aggregate_results
from kg_reasoning.workflows.state import (
    MultiAgentState,
    create_initial_state,
    update_state_from_orchestrator,
    update_state_from_preprocessor,
    update_state_from_worker,
    update_state_from_aggregator,
    update_state_from_synthesizer,
)

load_dotenv()


def preprocessor_node(state: MultiAgentState) -> MultiAgentState:
    """PreProcessor: expand query → search evidence → filter → resolve IDs + chunk_ids."""
    print("\n" + "=" * 60)
    print("🔍 PRE-PROCESSOR AGENT")
    print("=" * 60)

    try:
        state["current_step"] = "preprocessor_running"

        preprocessor = PreProcessor(
            llm_model=state.get("llm_model_preprocessor", PREPROCESSING_MODEL),
        )

        result = preprocessor.run(state["user_query"])
        state = update_state_from_preprocessor(state, result)

        print(f"\n  ✅ Pre-processor complete:")
        print(f"     - Query type:      {state['preprocessor_query_type']}")
        print(f"     - Entity IDs:      {len(state['preprocessor_entity_ids'])}")
        print(f"     - Predicate IDs:   {len(state['preprocessor_predicate_ids'])}")
        print(f"     - Community IDs:   {state['preprocessor_community_ids']}")
        print(f"     - Chunk IDs:       {len(state['preprocessor_chunk_ids'])}")
        chunk_texts = state.get("preprocessor_chunk_texts", {})
        fetched_count = sum(1 for v in chunk_texts.values() if v.get("text"))
        print(f"     - S3 chunk texts:  {fetched_count}/{len(state['preprocessor_chunk_ids'])} fetched")
        print(f"     - Needs pathing:   {state['preprocessor_needs_pathing']}")
        print(f"     - Needs community: {state['preprocessor_needs_community']}")

    except Exception as e:
        print(f"\n  ⚠️  Pre-processor failed: {e}")
        print("     Orchestrator will use original ReAct flow")
        state["current_step"] = "preprocessor_fallback"

    return state


def orchestrator_node(state: MultiAgentState) -> MultiAgentState:
    """Orchestrator: reads preprocessor signals → emits WorkerSpec list.

    Uses the fast single-shot structured-output path when the preprocessor
    resolved at least some canonical IDs or chunk IDs. Falls back to the
    original ReAct tool-calling loop when the preprocessor produced nothing.
    """
    print("\n" + "=" * 60)
    print("🎯 ORCHESTRATOR AGENT")
    print("=" * 60)

    try:
        state["current_step"] = "orchestrator_running"

        orchestrator = OrchestratorAgent(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model_orchestrator"],
        )

        has_preprocessor_output = (
            state.get("preprocessor_entity_ids")
            or state.get("preprocessor_predicate_ids")
            or state.get("preprocessor_chunk_ids")
        )

        if has_preprocessor_output:
            print("  → Using structured spec-planning (single-shot, no ReAct loop)")
            result = orchestrator.analyze_and_plan_with_context(state)
        else:
            print("  → Using ReAct fallback (preprocessor produced no output)")
            result = orchestrator.analyze_and_plan(state["user_query"])

        state = update_state_from_orchestrator(state, result)

        specs = state.get("worker_specs", [])
        print(f"\n  ✅ Orchestrator complete:")
        print(f"     - Entities found:      {len(state['entities_found'])}")
        print(f"     - Predicates found:    {len(state['predicates_found'])}")
        print(f"     - Communities:         {len(state['communities_identified'])}")
        print(f"     - Resolution method:   {state.get('resolution_method', 'unknown')}")
        print(f"     - Worker specs:        {len(specs)}")

        for i, spec in enumerate(specs, 1):
            tool = spec.get("tool") or spec.get("approach", "?")
            name = spec.get("name", "unnamed")
            print(f"       {i}. [{tool}] {name}")

    except Exception as e:
        print(f"\n  ❌ Orchestrator error: {e}")
        state["status"] = "error"
        state["error"] = f"Orchestrator failed: {e}"

    return state


def workers_node(state: MultiAgentState) -> MultiAgentState:
    """Workers: execute WorkerSpecs in parallel via TOOL_MAP, no LLM calls."""
    print("\n" + "=" * 60)
    print("⚙️  WORKER AGENTS (Parallel Execution)")
    print("=" * 60)

    try:
        state["current_step"] = "workers_running"

        # Accept both new worker_specs and legacy strategies key
        specs = state.get("worker_specs") or state.get("strategies", [])

        if not specs:
            print("  ⚠️  No specs to execute")
            return state

        print(f"\n  Executing {len(specs)} specs in parallel...")

        results = []
        with ThreadPoolExecutor(max_workers=min(len(specs), 5)) as executor:
            future_to_spec = {
                executor.submit(
                    execute_strategy_parallel,
                    spec,
                    state["user_query"],
                ): spec
                for spec in specs
            }

            for i, future in enumerate(as_completed(future_to_spec), 1):
                spec = future_to_spec[future]
                tool = spec.get("tool") or spec.get("approach", "?")
                name = spec.get("name", "unnamed")
                try:
                    result = future.result()
                    results.append(result)
                    status = result.get("status", "unknown")
                    count = result.get("results_count", 0)
                    print(f"     ✓ [{i}/{len(specs)}] [{tool}] {name} — {status} ({count} results)")
                except Exception as e:
                    print(f"     ✗ [{i}/{len(specs)}] [{tool}] {name} — Error: {e}")
                    results.append({
                        "strategy_name": name,
                        "status": "error",
                        "error": str(e),
                        "markdown_file": None,
                        "results_count": 0,
                    })

        for result in results:
            state = update_state_from_worker(state, result)

        successful = [r for r in results if r.get("status") == "success"]
        print(f"\n  ✅ Workers complete:")
        print(f"     - Successful:    {len(successful)}/{len(results)}")
        print(f"     - Total results: {state.get('total_results_count', 0)}")
        print(f"     - Files written: {len(state.get('markdown_files', []))}")

    except Exception as e:
        print(f"\n  ❌ Workers error: {e}")
        state["status"] = "error"
        state["error"] = f"Workers failed: {e}"

    return state


def aggregator_node(state: MultiAgentState) -> MultiAgentState:
    """Aggregator: deduplicate + rank worker outputs → single file for synthesizer.

    NEW NODE. Pure Python — no LLM, no network call, just file I/O + string
    scoring. Adds no meaningful latency but saves the synthesizer from
    reading N files and doing dedup/ranking itself inside its context window.
    """
    print("\n" + "=" * 60)
    print("📊 AGGREGATOR")
    print("=" * 60)

    try:
        state["current_step"] = "aggregator_running"

        markdown_files = state.get("markdown_files", [])

        if not markdown_files:
            print("  ⚠️  No markdown files to aggregate — synthesizer will report no results")
            state["aggregated_filepath"] = None
            state["aggregated_result_count"] = 0
            state["current_step"] = "aggregator_complete"
            return state

        result = aggregate_results(
            markdown_files=markdown_files,
            user_query=state["user_query"],
            entity_details=state.get("preprocessor_entity_details", {}),
            predicate_details=state.get("preprocessor_predicate_details", {}),
            chunk_texts=state.get("preprocessor_chunk_texts", {}),
            community_labels=state.get("preprocessor_community_labels", {}),
            # NEW: chunk_id -> {score, community_id, quote}, where "score" is
            # the REAL cosine similarity between the user's query and that
            # chunk's embedding (computed by the preprocessor's own evidence
            # search — see preprocessor._restructure_by_indices). This was
            # already sitting in state (state.py populates it from the
            # preprocessor's output) but was never threaded into the
            # aggregator call, so every triple/path was scored on lexical
            # overlap alone even though a real embedding-similarity score
            # existed for it the whole time. Without this line,
            # aggregate_results silently falls back to lexical-only scoring
            # (it warns but does not fail) — passing it is what actually
            # turns on embedding-aware reranking.
            chunk_details=state.get("preprocessor_chunk_details", {}),
        )
        state = update_state_from_aggregator(state, result)

        chunk_details_count = len(state.get("preprocessor_chunk_details", {}) or {})
        print(f"\n  ✅ Aggregator complete:")
        print(f"     - Input files:    {len(markdown_files)}")
        print(f"     - Unique results: {state['aggregated_result_count']}")
        print(f"     - Output file:    {state.get('aggregated_filepath', 'none')}")
        print(f"     - Chunk scores:   {chunk_details_count} available for embedding-based ranking"
              if chunk_details_count else
              "     - Chunk scores:   0 available — ranking fell back to lexical overlap only")

    except Exception as e:
        print(f"\n  ⚠️  Aggregator error: {e} — synthesizer will fall back to reading raw files")
        state["current_step"] = "aggregator_fallback"

    return state


def synthesizer_node(state: MultiAgentState) -> MultiAgentState:
    """Synthesizer: single LLM call against the aggregated file — no ReAct loop."""
    print("\n" + "=" * 60)
    print("📝 SYNTHESIZER AGENT")
    print("=" * 60)

    try:
        state["current_step"] = "synthesizer_running"

        synthesizer = SynthesizerAgent(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model_synthesizer"],
        )

        # Pass the pre-ranked single file when available; synthesizer falls
        # back to reading all recent files if the aggregator node failed.
        result = synthesizer.synthesize_answer(
            user_query=state["user_query"],
            strategies=state.get("worker_specs") or state.get("strategies"),
            aggregated_filepath=state.get("aggregated_filepath"),
        )

        state = update_state_from_synthesizer(state, result)

        print(f"\n  ✅ Synthesizer complete:")
        print(f"     - Files read:      {state['files_read']}")
        print(f"     - Results analyzed:{state['results_analyzed']}")
        print(f"     - Answer quality:  {state['synthesis_quality']}")

    except Exception as e:
        print(f"\n  ❌ Synthesizer error: {e}")
        state["status"] = "error"
        state["error"] = f"Synthesizer failed: {e}"

    return state


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def should_continue_after_orchestrator(state: MultiAgentState) -> str:
    if state.get("status") == "error":
        return END

    specs = state.get("worker_specs") or state.get("strategies", [])
    if not specs:
        print("  ⚠️  No specs planned — skipping workers and aggregator")
        return "synthesizer"

    return "workers"


def should_continue_after_workers(state: MultiAgentState) -> str:
    if state.get("status") == "error":
        return END
    return "aggregator"


def should_continue_after_aggregator(state: MultiAgentState) -> str:
    if state.get("status") == "error":
        return END
    return "synthesizer"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_workflow() -> StateGraph:
    """Create the multi-agent LangGraph workflow.

    Node order:
      preprocessor → orchestrator → workers → aggregator → synthesizer → END
    """
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("workers", workers_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("preprocessor")
    workflow.add_edge("preprocessor", "orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {"workers": "workers", "synthesizer": "synthesizer", END: END},
    )
    workflow.add_conditional_edges(
        "workers",
        should_continue_after_workers,
        {"aggregator": "aggregator", END: END},
    )
    workflow.add_conditional_edges(
        "aggregator",
        should_continue_after_aggregator,
        {"synthesizer": "synthesizer", END: END},
    )
    workflow.add_edge("synthesizer", END)

    return workflow


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_multi_agent_workflow(
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
) -> Dict[str, Any]:
    """Run the complete multi-agent reasoning workflow.

    Args:
        user_query: The user's question
        llm_provider: LLM provider to use
        llm_model_orchestrator: Model for orchestrator (single structured-output call)
        llm_model_synthesizer: Model for synthesizer (single non-agentic call)
        llm_model_preprocessor: Model for pre-processor (expand + filter calls)
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        neo4j_uri: Neo4j server URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Final state dictionary with answer and metadata
    """
    print("\n" + "=" * 60)
    print("🚀 MULTI-AGENT KG REASONING WORKFLOW  v2.0")
    print("=" * 60)
    print(f"\n  Query: {user_query}\n")

    initial_state = create_initial_state(
        user_query=user_query,
        llm_provider=llm_provider,
        llm_model_orchestrator=llm_model_orchestrator,
        llm_model_synthesizer=llm_model_synthesizer,
        llm_model_preprocessor=llm_model_preprocessor,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    workflow = create_workflow()
    app = workflow.compile()
    final_state = app.invoke(initial_state)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 WORKFLOW SUMMARY")
    print("=" * 60)
    status = final_state.get("status", "unknown")
    elapsed = final_state.get("execution_time_seconds") or 0
    specs = final_state.get("worker_specs") or final_state.get("strategies", [])
    unique_results = final_state.get("aggregated_result_count", 0)
    total_results = final_state.get("total_results_count", 0)

    print(f"  Status:           {status}")
    print(f"  Execution time:   {elapsed:.2f}s")
    print(f"  Specs executed:   {len(specs)}")
    print(f"  Total results:    {total_results}")
    print(f"  Unique (deduped): {unique_results}")
    print(f"  Markdown files:   {len(final_state.get('markdown_files', []))}")
    print(f"  Aggregated file:  {final_state.get('aggregated_filepath', 'none')}")

    if final_state.get("error"):
        print(f"\n  ❌ Error: {final_state['error']}")
    else:
        quality = final_state.get("synthesis_quality", "unknown")
        print(f"\n  ✅ Answer generated (quality: {quality})")

    print("=" * 60 + "\n")

    return dict(final_state)