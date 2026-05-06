"""Worker agent for multi-agent reasoning system.

Takes query strategies from the orchestrator and executes them directly on
Neo4j without any LLM round-trips — approach type determines which tool runs.
"""

from typing import Any, Dict, List, Optional

from kg_reasoning.agents.tools.neo4j_tools import _get_manager as _get_neo4j_manager
from kg_reasoning.agents.tools.markdown_tools import _get_manager as _get_md_manager


def _execute_expansion(
    entity_ids: List[str],
    community_ids: List[str],
    parameters: Dict[str, Any],
    strategy_name: str,
) -> str:
    """Run graph expansion from entity IDs and save results."""
    neo = _get_neo4j_manager()
    md = _get_md_manager()

    depth = parameters.get("depth", 1)
    limit = parameters.get("limit", 50)
    rel_types = parameters.get("relationship_types") or None

    # Build expansion query
    rel_filter = f":{':'.join(rel_types)}" if rel_types else ""
    if depth == 1:
        pattern = f"(start)-[r{rel_filter}]-(connected)"
        return_clause = "RETURN start, r, connected, r.community_id AS community_id"
    else:
        pattern = f"path = (start)-[r{rel_filter}*1..{depth}]-(connected)"
        return_clause = "RETURN path, relationships(path) AS rels"

    community_filter = ""
    params: Dict[str, Any] = {"entity_ids": entity_ids}
    if community_ids:
        community_filter = "AND r.community_id IN $community_ids"
        params["community_ids"] = community_ids

    cypher = f"""MATCH (start)
WHERE start.canonical_id IN $entity_ids
MATCH {pattern}
WHERE true {community_filter}
{return_clause}
LIMIT {limit}"""

    results = neo.execute_cypher_query(cypher, params, limit)
    filepath = md.write_query_results(strategy_name, cypher, results)
    return filepath


def _execute_direct(
    entity_ids: List[str],
    predicate_ids: List[str],
    community_ids: List[str],
    parameters: Dict[str, Any],
    strategy_name: str,
) -> str:
    """Run direct triple lookup by canonical IDs and save results."""
    neo = _get_neo4j_manager()
    md = _get_md_manager()

    limit = parameters.get("limit", 50)

    conditions = []
    params: Dict[str, Any] = {}

    if entity_ids:
        conditions.append("(subject.canonical_id IN $entity_ids OR object.canonical_id IN $entity_ids)")
        params["entity_ids"] = entity_ids
    if predicate_ids:
        conditions.append("r.canonical_id IN $predicate_ids")
        params["predicate_ids"] = predicate_ids
    if community_ids:
        conditions.append("r.community_id IN $community_ids")
        params["community_ids"] = community_ids

    where = " AND ".join(conditions) if conditions else "true"

    cypher = f"""MATCH (subject)-[r]->(object)
WHERE {where}
RETURN subject, r, object, r.community_id AS community_id
LIMIT {limit}"""

    results = neo.execute_cypher_query(cypher, params, limit)
    filepath = md.write_query_results(strategy_name, cypher, results)
    return filepath


def _execute_path(
    entity_ids: List[str],
    community_ids: List[str],
    parameters: Dict[str, Any],
    strategy_name: str,
) -> str:
    """Run shortest-path search between entity pairs and save results."""
    neo = _get_neo4j_manager()
    md = _get_md_manager()

    limit = parameters.get("limit", 20)
    max_depth = parameters.get("depth", 3)

    params: Dict[str, Any] = {"entity_ids": entity_ids}
    community_filter = ""
    if community_ids:
        community_filter = "WHERE ALL(r IN relationships(p) WHERE r.community_id IN $community_ids)"
        params["community_ids"] = community_ids

    cypher = f"""MATCH (a), (b)
WHERE a.canonical_id IN $entity_ids
  AND b.canonical_id IN $entity_ids
  AND a <> b
MATCH p = shortestPath((a)-[*1..{max_depth}]-(b))
{community_filter}
RETURN p
LIMIT {limit}"""

    results = neo.execute_cypher_query(cypher, params, limit)
    filepath = md.write_query_results(strategy_name, cypher, results)
    return filepath


class WorkerAgent:
    """Executes a single query strategy directly — no LLM round-trips."""

    def execute_strategy(
        self,
        strategy: Dict[str, Any],
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a query strategy by dispatching on approach type.

        Args:
            strategy: Strategy dictionary from orchestrator
            user_query: Original user query (unused, kept for API compatibility)

        Returns:
            Dictionary with execution results and metadata
        """
        name = strategy.get("name", "unnamed")
        approach = strategy.get("approach", "direct")
        canonical_ids = strategy.get("canonical_ids", {})
        entity_ids: List[str] = canonical_ids.get("entities", [])
        predicate_ids: List[str] = canonical_ids.get("predicates", [])
        community_ids: List[str] = strategy.get("community_ids", [])
        parameters: Dict[str, Any] = strategy.get("parameters", {})

        try:
            if approach == "expansion":
                filepath = _execute_expansion(entity_ids, community_ids, parameters, name)
            elif approach == "path":
                filepath = _execute_path(entity_ids, community_ids, parameters, name)
            else:  # "direct" and anything else
                filepath = _execute_direct(entity_ids, predicate_ids, community_ids, parameters, name)

            return {
                "strategy_name": name,
                "status": "success",
                "markdown_file": filepath,
                "output": f"Results written to: {filepath}",
            }

        except Exception as e:
            return {
                "strategy_name": name,
                "status": "error",
                "error": str(e),
                "markdown_file": None,
                "output": f"Error: {e}",
            }


def execute_strategy_parallel(
    strategy: Dict[str, Any],
    user_query: Optional[str] = None,
    llm_model: str = None,
) -> Dict[str, Any]:
    """Helper for parallel execution — llm_model unused, kept for API compatibility."""
    return WorkerAgent().execute_strategy(strategy, user_query)
