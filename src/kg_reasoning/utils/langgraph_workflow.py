"""LangGraph-based workflow for knowledge graph reasoning and query processing."""

import os
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from kg_reasoning.utils.entity_extractor import EntityExtractor
from kg_reasoning.utils.cypher_generator import CypherGenerator
from kg_reasoning.utils.neo4j_query import Neo4jQuery
from kg_reasoning.utils.answer_synthesizer import AnswerSynthesizer

# Load environment variables
load_dotenv()


class ReasoningState(TypedDict):
    """State for the knowledge graph reasoning workflow."""

    user_query: str
    llm_provider: str
    llm_model: str
    similarity_threshold: float

    # Qdrant configuration
    qdrant_url: str | None
    qdrant_api_key: str | None

    # Neo4j configuration
    neo4j_uri: str | None
    neo4j_user: str | None
    neo4j_password: str | None

    # Intermediate results
    extracted_entities: Dict[str, Any] | None
    entity_matches: list[Dict[str, Any]] | None
    predicate_matches: list[Dict[str, Any]] | None
    keywords: list[str] | None
    refined_query: str | None
    cypher_query: str | None
    cypher_explanation: str | None
    neo4j_results: list[Dict[str, Any]] | None
    high_connectivity_nodes: list[Dict[str, Any]] | None
    query_suggestions: list[Dict[str, Any]] | None
    topics: list[str] | None

    # Final results
    answer: str | None

    # Status
    status: str
    error: str | None
    current_step: str


def extract_entities_node(state: ReasoningState) -> ReasoningState:
    """Extract entities and predicates from the user query using hybrid matching."""
    try:
        print("🔍 Extracting entities and predicates from query...")

        extractor = EntityExtractor(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model"],
            qdrant_url=state["qdrant_url"],
            qdrant_api_key=state["qdrant_api_key"],
            keyword_weight=0.7,  # Prioritize precision
            semantic_weight=0.3,  # But allow semantic flexibility
        )

        extraction_result = extractor.extract_entities(state["user_query"])

        state["extracted_entities"] = extraction_result
        state["current_step"] = "check_matches"

        entities_count = len(extraction_result.get("entities", []))
        predicates_count = len(extraction_result.get("predicates", []))
        keywords_count = len(extraction_result.get("keywords", []))
        print(f"✅ Entity extraction completed: {keywords_count} keywords, {entities_count} entities and {predicates_count} predicates found")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Entity extraction failed: {e}"
        print(f"❌ Error: {e}")

    return state


def check_matches_node(state: ReasoningState) -> ReasoningState:
    """Check if we have entity or predicate matches and determine next step."""
    try:
        if not state["extracted_entities"]:
            raise ValueError("No extracted entities available")

        # Get matches from the extraction result
        entity_matches = state["extracted_entities"].get("qdrant_entity_matches", [])
        predicate_matches = state["extracted_entities"].get("qdrant_predicate_matches", [])
        keywords = state["extracted_entities"].get("keywords", [])

        # Store matches in state
        state["entity_matches"] = entity_matches
        state["predicate_matches"] = predicate_matches
        state["keywords"] = keywords

        # Show match type breakdown
        keyword_entity_matches = [m for m in entity_matches if "keyword" in m.get("match_type", [])]
        semantic_entity_matches = [m for m in entity_matches if "semantic" in m.get("match_type", [])]

        if entity_matches or predicate_matches:
            state["current_step"] = "refine_query"
            print(f"✅ Hybrid matching found {len(entity_matches)} entity matches ({len(keyword_entity_matches)} keyword, {len(semantic_entity_matches)} semantic) and {len(predicate_matches)} predicate matches")
        else:
            state["current_step"] = "get_high_connectivity"
            print("ℹ️  No entity or predicate matches found, getting high connectivity nodes")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Match checking failed: {e}"
        print(f"❌ Error: {e}")

    return state


def refine_query_node(state: ReasoningState) -> ReasoningState:
    """Refine user query with canonical entities and predicates."""
    try:
        print("✏️  Refining query with canonical entities and predicates...")

        entity_matches = state.get("entity_matches", [])
        predicate_matches = state.get("predicate_matches", [])

        if not entity_matches and not predicate_matches:
            raise ValueError("No entity or predicate matches available")

        # Simple refinement: replace entities and predicates in the query
        refined_query = state["user_query"]

        # Replace entities
        for match in entity_matches:
            original = match["payload"].get("name", "")
            canonical = match["payload"].get("name", "")  # Use the matched name as canonical
            if original and canonical:
                refined_query = refined_query.replace(original, canonical)

        # Replace predicates
        for match in predicate_matches:
            original = match["payload"].get("name", "")
            canonical = match["payload"].get("name", "")  # Use the matched name as canonical
            if original and canonical:
                refined_query = refined_query.replace(original, canonical)

        state["refined_query"] = refined_query
        state["current_step"] = "generate_cypher"
        print(f"✅ Query refinement completed: {refined_query}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Query refinement failed: {e}"
        print(f"❌ Error: {e}")

    return state


def generate_cypher_node(state: ReasoningState) -> ReasoningState:
    """Generate Cypher query from refined query or high connectivity nodes."""
    try:
        print("🔮 Generating Cypher query...")

        generator = CypherGenerator(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model"],
        )

        # Check if we have high connectivity nodes (no entity matches case)
        if state.get("high_connectivity_nodes"):
            print("   Using high connectivity nodes for query generation")
            # Generate a query based on high connectivity nodes
            high_connectivity_nodes = state["high_connectivity_nodes"]
            node_names = [node.get("name", "") for node in high_connectivity_nodes if node.get("name")]

            # Create a query that explores relationships between high connectivity nodes
            if node_names:
                # Generate a query that asks about the data in general
                query_context = f"Available entities in the knowledge graph: {', '.join(node_names[:10])}"
                cypher_result = generator.generate_cypher(
                    f"{state['user_query']}. {query_context}. Please provide a comprehensive overview of what information is available."
                )
            else:
                # Fallback to general query
                cypher_result = generator.generate_cypher(
                    f"{state['user_query']}. Please provide a comprehensive overview of the available information."
                )
        elif state.get("refined_query"):
            # Normal case with refined query
            cypher_result = generator.generate_cypher(state["refined_query"])
        else:
            # Fallback to original query
            cypher_result = generator.generate_cypher(state["user_query"])

        state["cypher_query"] = cypher_result["cypher_query"]
        state["cypher_explanation"] = cypher_result["query_explanation"]
        state["current_step"] = "execute_query"
        print(f"✅ Cypher generation completed: {state['cypher_query']}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Cypher generation failed: {e}"
        print(f"❌ Error: {e}")

    return state


def execute_query_node(state: ReasoningState) -> ReasoningState:
    """Execute Cypher query against Neo4j."""
    try:
        print("🕸️  Executing Neo4j query...")
        print(f"   Query: {state['cypher_query']}")

        if not state["cypher_query"]:
            raise ValueError("No Cypher query available")

        neo4j = Neo4jQuery(
            neo4j_uri=state["neo4j_uri"],
            neo4j_user=state["neo4j_user"],
            neo4j_password=state["neo4j_password"],
        )

        results = neo4j.execute_query(state["cypher_query"])

        state["neo4j_results"] = results
        state["current_step"] = "synthesize_answer"
        print(f"✅ Query execution completed: {len(results)} results found")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Query execution failed: {e}"
        print(f"❌ Error: {e}")

    return state


def synthesize_answer_node(state: ReasoningState) -> ReasoningState:
    """Synthesize natural language answer from Neo4j results."""
    try:
        print("📝 Synthesizing answer...")

        if not state["neo4j_results"]:
            # Handle empty results gracefully
            state["answer"] = "No results found in the knowledge graph for your query. Please try rephrasing your question or ask about different entities."
            state["current_step"] = "complete"
            state["status"] = "success"
            print(f"✅ Answer synthesis completed (no results)")
            return state

        synthesizer = AnswerSynthesizer(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model"],
        )

        answer = synthesizer.synthesize_answer(
            state["user_query"],
            state["neo4j_results"],
            state["cypher_query"] or "",
        )

        state["answer"] = answer
        state["current_step"] = "complete"
        state["status"] = "success"
        print(f"✅ Answer synthesis completed")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Answer synthesis failed: {e}"
        print(f"❌ Error: {e}")

    return state


def get_high_connectivity_node(state: ReasoningState) -> ReasoningState:
    """Get high connectivity nodes from Neo4j for query suggestions."""
    try:
        print("📊 Getting high connectivity nodes...")

        neo4j = Neo4jQuery(
            neo4j_uri=state["neo4j_uri"],
            neo4j_user=state["neo4j_user"],
            neo4j_password=state["neo4j_password"],
        )

        nodes = neo4j.get_high_connectivity_nodes(limit=20)

        state["high_connectivity_nodes"] = nodes
        state["current_step"] = "generate_cypher"
        print(f"✅ High connectivity nodes retrieved: {len(nodes)} nodes")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"High connectivity retrieval failed: {e}"
        print(f"❌ Error: {e}")

    return state


def generate_suggestions_node(state: ReasoningState) -> ReasoningState:
    """Generate query suggestions based on high connectivity nodes."""
    try:
        print("💡 Generating query suggestions...")

        if not state["high_connectivity_nodes"]:
            raise ValueError("No high connectivity nodes available")

        synthesizer = AnswerSynthesizer(
            llm_provider=state["llm_provider"],
            llm_model=state["llm_model"],
        )

        suggestions = synthesizer.generate_query_suggestions(
            state["user_query"],
            state["high_connectivity_nodes"],
        )

        state["query_suggestions"] = suggestions.get("suggestions", [])
        state["topics"] = suggestions.get("topics", [])
        state["answer"] = suggestions.get("message", "No direct matches found. Please try rephrasing your question.")
        state["current_step"] = "complete"
        state["status"] = "success"
        print(f"✅ Query suggestions generated: {len(state['query_suggestions'])} suggestions")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Suggestion generation failed: {e}"
        print(f"❌ Error: {e}")

    return state


def should_refine_query(state: ReasoningState) -> str:
    """Determine whether to proceed to query refinement or get high connectivity nodes."""
    if state["status"] == "error":
        return "error"

    if state["current_step"] == "refine_query":
        return "refine_query"
    elif state["current_step"] == "get_high_connectivity":
        return "get_high_connectivity"
    else:
        return "error"


def create_langgraph_workflow() -> StateGraph:
    """Create a LangGraph workflow for knowledge graph reasoning."""

    # Create the workflow graph
    workflow = StateGraph(ReasoningState)

    # Add nodes
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("check_matches", check_matches_node)
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("generate_cypher", generate_cypher_node)
    workflow.add_node("execute_query", execute_query_node)
    workflow.add_node("synthesize_answer", synthesize_answer_node)
    workflow.add_node("get_high_connectivity", get_high_connectivity_node)
    workflow.add_node("generate_suggestions", generate_suggestions_node)

    # Define the edges
    workflow.set_entry_point("extract_entities")

    workflow.add_edge("extract_entities", "check_matches")

    # Conditional edge for query refinement vs high connectivity
    workflow.add_conditional_edges(
        "check_matches",
        should_refine_query,
        {
            "refine_query": "refine_query",
            "get_high_connectivity": "get_high_connectivity",
            "error": END,
        },
    )

    # Path with entity matches
    workflow.add_edge("refine_query", "generate_cypher")
    workflow.add_edge("generate_cypher", "execute_query")
    workflow.add_edge("execute_query", "synthesize_answer")
    workflow.add_edge("synthesize_answer", END)

    # Path without entity matches
    workflow.add_edge("get_high_connectivity", "generate_cypher")
    workflow.add_edge("generate_cypher", "execute_query")
    workflow.add_edge("execute_query", "synthesize_answer")
    workflow.add_edge("synthesize_answer", END)

    return workflow


def run_langgraph_workflow(
    user_query: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    similarity_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Run the LangGraph workflow for knowledge graph reasoning.

    Args:
        user_query: The user's natural language query
        llm_provider: LLM provider
        llm_model: LLM model
        similarity_threshold: Similarity threshold for entity matching

    Returns:
        Dictionary with reasoning results
    """
    # Create the workflow
    workflow = create_langgraph_workflow()
    app = workflow.compile()

    # Initialize state
    initial_state: ReasoningState = {
        "user_query": user_query,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "similarity_threshold": similarity_threshold,
        "qdrant_url": os.getenv("QDRANT_URL"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "neo4j_uri": os.getenv("NEO4J_URI"),
        "neo4j_user": os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
        "extracted_entities": None,
        "entity_matches": None,
        "predicate_matches": None,
        "keywords": None,
        "refined_query": None,
        "cypher_query": None,
        "cypher_explanation": None,
        "neo4j_results": None,
        "high_connectivity_nodes": None,
        "query_suggestions": None,
        "topics": None,
        "answer": None,
        "status": "in_progress",
        "error": None,
        "current_step": "extract_entities",
    }

    # Run the workflow
    print("🚀 Starting LangGraph reasoning workflow...")
    final_state = app.invoke(initial_state)

    # Return results
    return {
        "original_query": user_query,
        "status": final_state["status"],
        "answer": final_state["answer"],
        "metadata": {
            "entity_extraction": final_state["extracted_entities"],
            "entity_matches": final_state["entity_matches"],
            "predicate_matches": final_state["predicate_matches"],
            "keywords": final_state["keywords"],
            "refined_query": final_state["refined_query"],
            "cypher_query": final_state["cypher_query"],
            "cypher_explanation": final_state["cypher_explanation"],
            "neo4j_results": final_state["neo4j_results"],
            "high_connectivity_nodes": final_state["high_connectivity_nodes"],
            "suggestions": final_state["query_suggestions"],
            "topics": final_state["topics"],
        },
        "error": final_state["error"],
    }
