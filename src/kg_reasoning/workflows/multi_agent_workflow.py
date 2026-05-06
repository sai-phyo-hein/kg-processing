"""Multi-agent workflow using LangGraph.

Orchestrates the flow: Orchestrator → Workers (parallel) → Synthesizer
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from kg_extractor.utils.model_setup import (
    REASONING_PROVIDER,
    ORCHESTRATOR_MODEL,
    WORKER_MODEL,
    SYNTHESIZER_MODEL,
)
from kg_reasoning.agents.orchestrator import OrchestratorAgent
from kg_reasoning.agents.worker import WorkerAgent, execute_strategy_parallel
from kg_reasoning.agents.synthesizer import SynthesizerAgent
from kg_reasoning.workflows.state import (
    MultiAgentState,
    create_initial_state,
    update_state_from_orchestrator,
    update_state_from_worker,
    update_state_from_synthesizer,
)

load_dotenv()


def orchestrator_node(state: MultiAgentState) -> MultiAgentState:
    """Orchestrator node: Analyzes query and plans strategies.

    Args:
        state: Current workflow state

    Returns:
        Updated state with strategies
    """
    print("\n" + "=" * 60)
    print("🎯 ORCHESTRATOR AGENT")
    print("=" * 60)

    try:
        state["current_step"] = "orchestrator_running"

        # Initialize orchestrator
        orchestrator = OrchestratorAgent(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model_orchestrator"],
        )

        # Analyze and plan
        result = orchestrator.analyze_and_plan(state["user_query"])

        # Update state
        state = update_state_from_orchestrator(state, result)

        print(f"\n✅ Orchestrator complete:")
        print(f"   - Entities found: {len(state['entities_found'])}")
        print(f"   - Predicates found: {len(state['predicates_found'])}")
        print(f"   - Communities identified: {len(state['communities_identified'])}")
        print(f"   - Resolution method: {state.get('resolution_method', 'unknown')}")
        print(f"   - Strategies planned: {len(state['strategies'])}")

        for i, strategy in enumerate(state["strategies"], 1):
            print(f"   {i}. {strategy.get('name', 'unnamed')}")

    except Exception as e:
        print(f"\n❌ Orchestrator error: {e}")
        state["status"] = "error"
        state["error"] = f"Orchestrator failed: {e}"

    return state


def workers_node(state: MultiAgentState) -> MultiAgentState:
    """Workers node: Execute strategies in parallel.

    Args:
        state: Current workflow state

    Returns:
        Updated state with query results
    """
    print("\n" + "=" * 60)
    print("⚙️  WORKER AGENTS (Parallel Execution)")
    print("=" * 60)

    try:
        state["current_step"] = "workers_running"
        strategies = state.get("strategies", [])

        if not strategies:
            print("⚠️  No strategies to execute")
            return state

        print(f"\nExecuting {len(strategies)} strategies in parallel...")

        # Execute strategies in parallel using ThreadPoolExecutor
        results = []

        with ThreadPoolExecutor(max_workers=min(len(strategies), 5)) as executor:
            # Submit all strategies
            future_to_strategy = {
                executor.submit(
                    execute_strategy_parallel,
                    strategy,
                    state["user_query"],
                    state["llm_model_worker"],
                ): strategy
                for strategy in strategies
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_strategy), 1):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"   ✓ [{i}/{len(strategies)}] {strategy.get('name', 'unnamed')} - "
                        f"{result.get('status', 'unknown')}"
                    )
                except Exception as e:
                    print(f"   ✗ [{i}/{len(strategies)}] {strategy.get('name', 'unnamed')} - Error: {e}")
                    results.append({
                        "strategy_name": strategy.get("name", "unnamed"),
                        "status": "error",
                        "error": str(e),
                    })

        # Update state with all results
        for result in results:
            state = update_state_from_worker(state, result)

        print(f"\n✅ Workers complete:")
        print(f"   - Successful: {len([r for r in results if r.get('status') == 'success'])}")
        print(f"   - Failed: {len(state.get('failed_strategies', []))}")
        print(f"   - Total results: {state.get('total_results_count', 0)}")
        print(f"   - Markdown files: {len(state.get('markdown_files', []))}")

    except Exception as e:
        print(f"\n❌ Workers error: {e}")
        state["status"] = "error"
        state["error"] = f"Workers failed: {e}"

    return state


def synthesizer_node(state: MultiAgentState) -> MultiAgentState:
    """Synthesizer node: Generate final answer.

    Args:
        state: Current workflow state

    Returns:
        Updated state with final answer
    """
    print("\n" + "=" * 60)
    print("📝 SYNTHESIZER AGENT")
    print("=" * 60)

    try:
        state["current_step"] = "synthesizer_running"

        # Initialize synthesizer
        synthesizer = SynthesizerAgent(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model_synthesizer"],
        )

        # Synthesize answer
        result = synthesizer.synthesize_answer(
            state["user_query"],
            state.get("strategies", []),
        )

        # Update state
        state = update_state_from_synthesizer(state, result)

        print(f"\n✅ Synthesizer complete:")
        print(f"   - Files read: {state['files_read']}")
        print(f"   - Results analyzed: {state['results_analyzed']}")
        print(f"   - Answer quality: {state['synthesis_quality']}")

    except Exception as e:
        print(f"\n❌ Synthesizer error: {e}")
        state["status"] = "error"
        state["error"] = f"Synthesizer failed: {e}"

    return state


def should_continue_after_orchestrator(state: MultiAgentState) -> str:
    """Decision node after orchestrator.

    Args:
        state: Current workflow state

    Returns:
        Next node name or END
    """
    if state.get("status") == "error":
        return END

    if not state.get("strategies"):
        print("⚠️  No strategies planned, skipping to synthesizer")
        return "synthesizer"

    return "workers"


def should_continue_after_workers(state: MultiAgentState) -> str:
    """Decision node after workers.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    if state.get("status") == "error":
        return END

    return "synthesizer"


def create_workflow() -> StateGraph:
    """Create the multi-agent LangGraph workflow.

    Returns:
        Configured StateGraph
    """
    # Create graph
    workflow = StateGraph(MultiAgentState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("workers", workers_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Add edges
    workflow.set_entry_point("orchestrator")

    # Conditional edge after orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        should_continue_after_orchestrator,
        {
            "workers": "workers",
            "synthesizer": "synthesizer",
            END: END,
        },
    )

    # Conditional edge after workers
    workflow.add_conditional_edges(
        "workers",
        should_continue_after_workers,
        {
            "synthesizer": "synthesizer",
            END: END,
        },
    )

    # End after synthesizer
    workflow.add_edge("synthesizer", END)

    return workflow


def run_multi_agent_workflow(
    user_query: str,
    llm_provider: str = REASONING_PROVIDER,
    llm_model_orchestrator: str = ORCHESTRATOR_MODEL,
    llm_model_worker: str = WORKER_MODEL,
    llm_model_synthesizer: str = SYNTHESIZER_MODEL,
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
        llm_model_orchestrator: Model for orchestrator
        llm_model_worker: Model for workers
        llm_model_synthesizer: Model for synthesizer
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        neo4j_uri: Neo4j server URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password

    Returns:
        Final state dictionary with answer and metadata
    """
    print("\n" + "=" * 60)
    print("🚀 MULTI-AGENT KG REASONING WORKFLOW")
    print("=" * 60)
    print(f"\nUser Query: {user_query}\n")

    # Create initial state
    initial_state = create_initial_state(
        user_query=user_query,
        llm_provider=llm_provider,
        llm_model_orchestrator=llm_model_orchestrator,
        llm_model_worker=llm_model_worker,
        llm_model_synthesizer=llm_model_synthesizer,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )

    # Create and compile workflow
    workflow = create_workflow()
    app = workflow.compile()

    # Run workflow
    final_state = app.invoke(initial_state)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 WORKFLOW SUMMARY")
    print("=" * 60)
    print(f"Status: {final_state.get('status', 'unknown')}")
    print(f"Execution time: {final_state.get('execution_time_seconds') or 0:.2f}s")
    print(f"Strategies executed: {len(final_state.get('strategies', []))}")
    print(f"Total results: {final_state.get('total_results_count', 0)}")
    print(f"Markdown files created: {len(final_state.get('markdown_files', []))}")

    if final_state.get("error"):
        print(f"\n❌ Error: {final_state['error']}")
    else:
        print(f"\n✅ Answer generated (quality: {final_state.get('synthesis_quality', 'unknown')})")

    print("=" * 60 + "\n")

    return dict(final_state)
