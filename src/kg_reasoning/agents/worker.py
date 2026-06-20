"""Worker agent for multi-agent reasoning system.

Takes either a new-style WorkerSpec ({"tool": "graph_hop", ...}) or a
legacy strategy dict ({"approach": "expansion", "canonical_ids": {...}})
from the orchestrator and executes it directly — no LLM round-trips.

TOOL VOCABULARY (restricted, per orchestrator strategy):
Only three retrieval tools exist now — graph_hop ("neighborhoods"),
find_paths ("path between"), and community_explore ("large communities"
fallback). search_evidence, direct, fetch_chunks, surround_chunks,
fetch_s3_chunks, and surround_s3_chunks have been REMOVED entirely — not
deprioritized, not kept as dead code. The orchestrator (see
orchestrator.VALID_TOOLS) never emits specs for them, and the preprocessor's
own S3 fetch step (preprocessor.py Step 9) already retrieves chunk text
directly, so there is no chunk-level retrieval left for a worker to do.

TOOL_MAP is the single source of truth for "what can a worker actually do" —
it must stay in sync with orchestrator.VALID_TOOLS. It intentionally maps
both new tool names (graph_hop, find_paths, community_explore) and legacy
approach names (expansion, path, community_exploration) onto the same
executor functions, so the ReAct-fallback orchestrator path (which still
emits "approach" strings, now restricted to the same three) keeps working
without modification.
"""

from typing import Any, Callable, Dict, List, Optional

from kg_reasoning.agents.tools.neo4j_tools import _get_manager as _get_neo4j_manager
from kg_reasoning.agents.tools.markdown_tools import _get_manager as _get_md_manager


def _execute_expansion(
    entity_ids: List[str],
    community_ids: List[str],
    parameters: Dict[str, Any],
    strategy_name: str,
) -> Dict[str, Any]:
    """Run graph expansion from entity IDs and save results."""
    neo = _get_neo4j_manager()
    md = _get_md_manager()

    depth = parameters.get("depth", 1)
    limit = parameters.get("limit", 50)
    rel_types = parameters.get("relationship_types") or None

    rel_filter = f":{':'.join(rel_types)}" if rel_types else ""
    params: Dict[str, Any] = {"entity_ids": entity_ids}

    if depth == 1:
        pattern = f"(start)-[r{rel_filter}]-(connected)"
        return_clause = "RETURN start, r, connected, r.community_id AS community_id"

        community_filter = ""
        if community_ids:
            community_filter = "AND r.community_id IN $community_ids"
            params["community_ids"] = community_ids

        cypher = f"""MATCH (start)
WHERE start.canonical_id IN $entity_ids
MATCH {pattern}
WHERE true {community_filter}
{return_clause}
LIMIT {limit}"""
    else:
        pattern = f"path = (start)-[r{rel_filter}*1..{depth}]-(connected)"
        return_clause = "RETURN path, relationships(path) AS rels"

        community_filter = ""
        if community_ids:
            community_filter = "WHERE ALL(rel IN relationships(path) WHERE rel.community_id IN $community_ids)"
            params["community_ids"] = community_ids

        cypher = f"""MATCH (start)
WHERE start.canonical_id IN $entity_ids
MATCH {pattern}
{community_filter}
{return_clause}
LIMIT {limit}"""

    results = neo.execute_cypher_query(cypher, params, limit)
    filepath = md.write_query_results(strategy_name, cypher, results)
    return {"filepath": filepath, "result_count": len(results)}


def _execute_path(
    entity_ids: List[str],
    community_ids: List[str],
    parameters: Dict[str, Any],
    strategy_name: str,
) -> Dict[str, Any]:
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
    return {"filepath": filepath, "result_count": len(results)}


def _execute_community_exploration(
    community_ids: List[str],
    parameters: Dict[str, Any],
    strategy_name: str,
) -> Dict[str, Any]:
    """Run broad community exploration by querying all relationships in the community."""
    neo = _get_neo4j_manager()
    md = _get_md_manager()

    limit = parameters.get("limit", 100)
    params: Dict[str, Any] = {"community_ids": community_ids}

    cypher = f"""MATCH (subject)-[r]->(object)
WHERE r.community_id IN $community_ids
RETURN subject, r, object, r.community_id AS community_id
LIMIT {limit}"""

    results = neo.execute_cypher_query(cypher, params, limit)
    filepath = md.write_query_results(strategy_name, cypher, results)
    return {"filepath": filepath, "result_count": len(results)}


# ---------------------------------------------------------------------------
# TOOL_MAP — single source of truth for "what can a worker actually do".
# Keys cover both the new spec vocabulary (orchestrator.VALID_TOOLS) and the
# legacy "approach" strings the ReAct-fallback orchestrator path still
# emits, so both code paths dispatch through the same table.
# ---------------------------------------------------------------------------

def _dispatch_graph_hop(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    return _execute_expansion(
        spec.get("entity_ids", []),
        spec.get("community_ids", []),
        spec.get("parameters", {}),
        name,
    )


def _dispatch_find_paths(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    return _execute_path(
        spec.get("entity_ids", []),
        spec.get("community_ids", []),
        spec.get("parameters", {}),
        name,
    )


def _dispatch_community_explore(spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    return _execute_community_exploration(
        spec.get("community_ids", []),
        spec.get("parameters", {}),
        name,
    )


TOOL_MAP: Dict[str, Callable[[Dict[str, Any], str], Dict[str, Any]]] = {
    # New spec vocabulary
    "graph_hop": _dispatch_graph_hop,
    "find_paths": _dispatch_find_paths,
    "community_explore": _dispatch_community_explore,
    # Legacy "approach" strings — same executors, different entry key, so
    # the ReAct-fallback orchestrator path keeps working unmodified.
    "expansion": _dispatch_graph_hop,
    "path": _dispatch_find_paths,
    "community_exploration": _dispatch_community_explore,
}


class WorkerAgent:
    """Executes a single worker spec by dispatching through TOOL_MAP — no LLM round-trips."""

    def execute_strategy(
        self,
        strategy: Dict[str, Any],
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a worker spec (new-style) or legacy strategy dict.

        Accepts both shapes:
          New:    {"tool": "graph_hop", "entity_ids": [...], "parameters": {...}}
          Legacy: {"approach": "expansion", "canonical_ids": {"entities": [...]}, ...}

        Legacy dicts are normalized into the new shape before dispatch so
        there is exactly one execution path, not two parallel ones.

        Args:
            strategy: WorkerSpec or legacy strategy dict from the orchestrator
            user_query: Original user query (unused, kept for API compatibility)

        Returns:
            Dictionary with execution results and metadata
        """
        name = strategy.get("name", "unnamed")
        spec = self._normalize(strategy)
        tool = spec.get("tool", "community_explore")

        dispatch_fn = TOOL_MAP.get(tool, _dispatch_community_explore)

        try:
            result = dispatch_fn(spec, name)
            return {
                "strategy_name": name,
                "status": "success",
                "markdown_file": result["filepath"],
                "results_count": result["result_count"],
                "output": f"Results written to: {result['filepath']}",
            }
        except Exception as e:
            return {
                "strategy_name": name,
                "status": "error",
                "error": str(e),
                "markdown_file": None,
                "results_count": 0,
                "output": f"Error: {e}",
            }

    @staticmethod
    def _normalize(strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a legacy strategy dict into the new WorkerSpec shape.

        Legacy shape nests entity/predicate IDs under canonical_ids and
        uses "approach" instead of "tool". If the strategy already has a
        "tool" key, it's passed through unchanged.
        """
        if "tool" in strategy:
            return strategy

        canonical_ids = strategy.get("canonical_ids", {})
        return {
            "tool": strategy.get("approach", "community_explore"),
            "entity_ids": canonical_ids.get("entities", []),
            "predicate_ids": canonical_ids.get("predicates", []),
            "chunk_ids": strategy.get("chunk_ids", []),
            "community_ids": strategy.get("community_ids", []),
            "parameters": strategy.get("parameters", {}),
        }


def execute_strategy_parallel(
    strategy: Dict[str, Any],
    user_query: Optional[str] = None,
    llm_model: str = None,
) -> Dict[str, Any]:
    """Helper for parallel execution — llm_model unused, kept for API compatibility."""
    return WorkerAgent().execute_strategy(strategy, user_query)