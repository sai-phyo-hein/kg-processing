"""Neo4j tools for multi-agent reasoning system.

Provides LangChain tools for executing Cypher queries and expanding graph patterns
on Neo4j knowledge graphs.
"""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from neo4j import GraphDatabase

load_dotenv()


class Neo4jToolsManager:
    """Manager for Neo4j tools with shared driver and utilities."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize Neo4j tools manager.

        Args:
            neo4j_uri: Neo4j server URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        # Initialize driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
        )

    def close(self):
        """Close Neo4j driver."""
        if hasattr(self, "driver"):
            self.driver.close()

    def _serialize_value(self, value: Any) -> Any:
        """Serialize Neo4j value to JSON-serializable format.

        Args:
            value: Value to serialize

        Returns:
            JSON-serializable value
        """
        from neo4j.graph import Node, Relationship, Path
        from neo4j.time import DateTime, Date, Time, Duration

        if isinstance(value, Node):
            return {
                "type": "node",
                "id": value.id,
                "labels": list(value.labels),
                "properties": {k: self._serialize_value(v) for k, v in dict(value).items()},
            }
        elif isinstance(value, Relationship):
            return {
                "type": "relationship",
                "id": value.id,
                "type_name": value.type,
                "start_node": value.start_node.id,
                "end_node": value.end_node.id,
                "properties": {k: self._serialize_value(v) for k, v in dict(value).items()},
            }
        elif isinstance(value, Path):
            return {
                "type": "path",
                "nodes": [self._serialize_value(node) for node in value.nodes],
                "relationships": [self._serialize_value(rel) for rel in value.relationships],
            }
        elif isinstance(value, (DateTime, Date, Time)):
            # Convert Neo4j temporal types to ISO format strings
            return value.iso_format()
        elif isinstance(value, Duration):
            # Convert Neo4j duration to string representation
            return str(value)
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

    def execute_cypher_query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query on Neo4j.

        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            limit: Maximum number of results to return

        Returns:
            List of query results as dictionaries
        """
        if parameters is None:
            parameters = {}

        # Add limit if not in query
        if "LIMIT" not in cypher_query.upper():
            cypher_query = f"{cypher_query} LIMIT {limit}"

        results = []

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters)

                for record in result:
                    # Convert record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        record_dict[key] = self._serialize_value(value)

                    results.append(record_dict)

            return results

        except Exception as e:
            print(f"Error executing Cypher query: {e}")
            return [{"error": str(e), "query": cypher_query}]

    def expand_graph_query(
        self,
        entity_ids: List[str],
        relationship_types: Optional[List[str]] = None,
        community_id: Optional[str] = None,
        depth: int = 1,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand graph from entities to find connected nodes and relationships.

        Args:
            entity_ids: List of entity canonical IDs to start from
            relationship_types: Optional list of relationship types to filter
            community_id: Optional community ID to filter results
            depth: Depth of graph expansion (default: 1)
            limit: Maximum number of results to return

        Returns:
            List of expanded graph results
        """
        if not entity_ids:
            return []

        # Build Cypher query for graph expansion
        cypher_parts = []

        # Match starting entities
        cypher_parts.append(
            "MATCH (start) WHERE start.canonical_id IN $entity_ids"
        )

        # Build pattern for expansion
        rel_filter = ""
        if relationship_types:
            rel_filter = f":{':'.join(relationship_types)}"

        if depth == 1:
            cypher_parts.append(f"MATCH (start)-[r{rel_filter}]-(connected)")
        else:
            cypher_parts.append(
                f"MATCH path = (start)-[r{rel_filter}*1..{depth}]-(connected)"
            )

        # Add community filter if specified
        if community_id:
            cypher_parts.append(f"WHERE r.community_id = $community_id")

        # Return results
        if depth == 1:
            cypher_parts.append(
                "RETURN start, r, connected, "
                "r.community_id as community_id "
                f"LIMIT {limit}"
            )
        else:
            cypher_parts.append(
                "RETURN path, "
                "relationships(path) as rels "
                f"LIMIT {limit}"
            )

        cypher_query = "\n".join(cypher_parts)

        parameters = {"entity_ids": entity_ids}
        if community_id:
            parameters["community_id"] = community_id

        return self.execute_cypher_query(cypher_query, parameters, limit)

    def query_by_canonical_ids(
        self,
        subject_ids: Optional[List[str]] = None,
        predicate_ids: Optional[List[str]] = None,
        object_ids: Optional[List[str]] = None,
        community_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Query graph using canonical IDs for subjects, predicates, and objects.

        Args:
            subject_ids: Optional list of subject canonical IDs
            predicate_ids: Optional list of predicate canonical IDs
            object_ids: Optional list of object canonical IDs
            community_id: Optional community ID to filter
            limit: Maximum number of results

        Returns:
            List of matching triples
        """
        conditions = []
        parameters = {}

        # Build WHERE clause
        if subject_ids:
            conditions.append("subject.canonical_id IN $subject_ids")
            parameters["subject_ids"] = subject_ids

        if object_ids:
            conditions.append("object.canonical_id IN $object_ids")
            parameters["object_ids"] = object_ids

        if predicate_ids:
            conditions.append("r.canonical_id IN $predicate_ids")
            parameters["predicate_ids"] = predicate_ids

        if community_id:
            conditions.append("r.community_id = $community_id")
            parameters["community_id"] = community_id

        where_clause = " AND ".join(conditions) if conditions else "true"

        cypher_query = f"""
        MATCH (subject)-[r]->(object)
        WHERE {where_clause}
        RETURN subject, r, object, r.community_id as community_id
        LIMIT {limit}
        """

        return self.execute_cypher_query(cypher_query, parameters, limit)


# Global manager instance
_manager = None


def _get_manager() -> Neo4jToolsManager:
    """Get or create global Neo4jToolsManager instance."""
    global _manager
    if _manager is None:
        _manager = Neo4jToolsManager()
    return _manager


@tool
def execute_cypher_query(cypher_query: str, parameters: str = "{}") -> str:
    """Execute a Cypher query on the Neo4j knowledge graph.

    Use this tool to run custom Cypher queries to retrieve information from the graph.
    The query results are automatically serialized to JSON format.

    Args:
        cypher_query: The Cypher query string to execute
        parameters: JSON string of query parameters (default: "{}")

    Returns:
        JSON string with query results including nodes, relationships, and properties
    """
    manager = _get_manager()

    # Parse parameters
    try:
        params = json.loads(parameters) if parameters else {}
    except json.JSONDecodeError:
        params = {}

    results = manager.execute_cypher_query(cypher_query, params)
    return json.dumps(results, indent=2)


@tool
def expand_graph_query(
    entity_ids: str,
    relationship_types: str = "",
    community_id: str = "",
    depth: int = 1,
    limit: int = 50,
) -> str:
    """Expand graph from entities to discover connected nodes and relationships.

    Use this tool to explore the graph starting from known entities. It will find
    all connected nodes within the specified depth and optionally filter by
    relationship types and community ID.

    Args:
        entity_ids: Comma-separated list of entity canonical IDs to start from
        relationship_types: Comma-separated relationship types to filter (empty for all)
        community_id: Community ID to filter results (empty for all communities)
        depth: Depth of graph expansion, 1-3 (default: 1)
        limit: Maximum number of results to return (default: 50)

    Returns:
        JSON string with expanded graph including nodes, relationships, and paths
    """
    manager = _get_manager()

    # Parse inputs
    ids = [eid.strip() for eid in entity_ids.split(",") if eid.strip()]
    rel_types = (
        [rt.strip() for rt in relationship_types.split(",") if rt.strip()]
        if relationship_types
        else None
    )
    comm_id = community_id if community_id else None

    results = manager.expand_graph_query(ids, rel_types, comm_id, depth, limit)
    return json.dumps(results, indent=2)


@tool
def query_by_canonical_ids(
    subject_ids: str = "",
    predicate_ids: str = "",
    object_ids: str = "",
    community_id: str = "",
    limit: int = 50,
) -> str:
    """Query graph using canonical IDs from Qdrant registries.

    Use this tool to find triples matching specific subjects, predicates, and objects
    by their canonical IDs. This is the most precise way to query the graph after
    getting canonical IDs from Qdrant.

    Args:
        subject_ids: Comma-separated subject canonical IDs (empty for any)
        predicate_ids: Comma-separated predicate canonical IDs (empty for any)
        object_ids: Comma-separated object canonical IDs (empty for any)
        community_id: Community ID to filter results (empty for all communities)
        limit: Maximum number of results to return (default: 50)

    Returns:
        JSON string with matching triples including subjects, predicates, objects
    """
    manager = _get_manager()

    # Parse inputs
    subj_ids = [sid.strip() for sid in subject_ids.split(",") if sid.strip()] if subject_ids else None
    pred_ids = [pid.strip() for pid in predicate_ids.split(",") if pid.strip()] if predicate_ids else None
    obj_ids = [oid.strip() for oid in object_ids.split(",") if oid.strip()] if object_ids else None
    comm_id = community_id if community_id else None

    results = manager.query_by_canonical_ids(subj_ids, pred_ids, obj_ids, comm_id, limit)
    return json.dumps(results, indent=2)


@tool
def execute_and_save_query(
    cypher_query: str,
    strategy_name: str,
    parameters: str = "{}",
) -> str:
    """Execute a Cypher query and immediately save results to a markdown file.

    Use this combined tool instead of calling execute_cypher_query and
    write_query_results separately. It avoids passing large JSON blobs between
    tool calls, which can cause parse errors.

    Args:
        cypher_query: The Cypher query string to execute
        strategy_name: Name describing the query strategy (used in the filename)
        parameters: JSON string of query parameters (default: "{}")

    Returns:
        Path to the saved results markdown file
    """
    from kg_reasoning.agents.tools.markdown_tools import _get_manager as _get_md_manager

    neo_manager = _get_manager()
    md_manager = _get_md_manager()

    try:
        params = json.loads(parameters) if parameters else {}
    except json.JSONDecodeError:
        params = {}

    results = neo_manager.execute_cypher_query(cypher_query, params)
    filepath = md_manager.write_query_results(strategy_name, cypher_query, results)
    return f"Results written to: {filepath}"


@tool
def get_top_connected_nodes(limit: int = 20) -> str:
    """Get the most connected nodes in the knowledge graph by relationship count.

    Use this tool as a fallback when no canonical entities or predicates were found
    for the user's query. The top connected nodes act as high-value entry points
    for building exploratory query strategies.

    Args:
        limit: Number of top nodes to return (default: 20)

    Returns:
        JSON string with nodes sorted by descending degree, including their
        labels, canonical_id, name, and relationship count.
    """
    manager = _get_manager()

    cypher_query = f"""
    MATCH (n)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) AS degree
    ORDER BY degree DESC
    LIMIT {limit}
    RETURN
        n.canonical_id   AS canonical_id,
        n.name           AS name,
        labels(n)        AS labels,
        degree
    """

    results = manager.execute_cypher_query(cypher_query, {}, limit)
    return json.dumps(results, indent=2)


@tool
def get_community_ids_from_relationships() -> str:
    """Get all unique community IDs from relationship properties in Neo4j.

    Use this tool to discover what communities exist in the knowledge graph.
    Community IDs are stored in relationship properties and can be used to
    filter queries to specific communities.

    Returns:
        JSON string with list of unique community IDs found in the graph
    """
    manager = _get_manager()

    cypher_query = """
    MATCH ()-[r]->()
    WHERE r.community_id IS NOT NULL
    RETURN DISTINCT r.community_id AS community_id
    ORDER BY community_id
    """

    results = manager.execute_cypher_query(cypher_query, {}, 1000)
    
    # Extract just the community_id strings
    community_ids = [r.get("community_id") for r in results if r.get("community_id")]
    
    return json.dumps(community_ids, indent=2)
