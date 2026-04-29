"""Neo4j query module for knowledge graph querying."""

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Neo4jQuery:
    """Query Neo4j knowledge graph."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize the Neo4j query client.

        Args:
            neo4j_uri: Neo4j server URI (from env if not provided)
            neo4j_user: Neo4j username (from env if not provided)
            neo4j_password: Neo4j password (from env if not provided)
        """
        import os

        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

        # Initialize Neo4j driver
        self._init_neo4j_driver()

    def _init_neo4j_driver(self):
        """Initialize Neo4j driver."""
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            print(f"Connected to Neo4j at {self.neo4j_uri}")
        except ImportError:
            raise ImportError(
                "neo4j is required. Install with: pip install neo4j"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Neo4j driver: {e}")

    def close(self):
        """Close the Neo4j driver."""
        if hasattr(self, 'driver'):
            self.driver.close()

    def _serialize_record(self, record) -> Dict[str, Any]:
        """Serialize a Neo4j record to JSON-serializable dictionary.

        Args:
            record: Neo4j Record object

        Returns:
            JSON-serializable dictionary
        """
        result = {}
        for key, value in record.items():
            result[key] = self._serialize_value(value)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value to JSON-serializable format.

        Args:
            value: Value to serialize

        Returns:
            JSON-serializable value
        """
        # Check if it's a Node object by checking for common Node attributes
        if hasattr(value, 'element_id') and hasattr(value, 'labels') and hasattr(value, 'items'):
            # It's a Node object
            return {
                "id": value.element_id,
                "labels": list(value.labels),
                "properties": {k: self._serialize_value(v) for k, v in value.items()}
            }
        # Check if it's a Relationship object
        elif hasattr(value, 'element_id') and hasattr(value, 'type') and hasattr(value, 'start_node'):
            # It's a Relationship object
            return {
                "id": value.element_id,
                "type": value.type,
                "start_node": value.start_node.element_id if hasattr(value.start_node, 'element_id') else str(value.start_node),
                "end_node": value.end_node.element_id if hasattr(value.end_node, 'element_id') else str(value.end_node),
                "properties": {k: self._serialize_value(v) for k, v in value.items()}
            }
        # Check if it's a datetime object
        elif hasattr(value, 'isoformat'):
            # It's likely a datetime or similar object
            return value.isoformat()
        elif isinstance(value, list):
            # Handle lists recursively
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            # Handle dictionaries recursively
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Basic types that are JSON-serializable
            return value
        else:
            # Convert to string for other types
            return str(value)

    def execute_query(
        self,
        cypher_query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            cypher_query: The Cypher query to execute
            params: Optional parameters for the query

        Returns:
            List of result dictionaries
        """
        results = []

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params or {})

                for record in result:
                    # Convert Record to JSON-serializable dictionary
                    results.append(self._serialize_record(record))

        except Exception as e:
            print(f"Error executing Neo4j query: {e}")

        return results

    def get_high_connectivity_nodes(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get nodes with the highest connectivity.

        Args:
            limit: Maximum number of nodes to return

        Returns:
            List of nodes with their connection counts
        """
        cypher_query = """
        MATCH (n:Entity)-[r]-()
        WITH n, count(r) as connections
        RETURN n.name as name, n.type as type, n.canonical_id as canonical_id, connections
        ORDER BY connections DESC
        LIMIT $limit
        """

        results = self.execute_query(cypher_query, {"limit": limit})

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "name": result.get("name", ""),
                "type": result.get("type", ""),
                "canonical_id": result.get("canonical_id", ""),
                "connection_count": result.get("connections", 0),
            })

        return formatted_results

    def get_node_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a node by its name.

        Args:
            name: The name of the node

        Returns:
            Node dictionary or None if not found
        """
        cypher_query = """
        MATCH (n:Entity {name: $name})
        RETURN n
        LIMIT 1
        """

        results = self.execute_query(cypher_query, {"name": name})

        if results:
            node = results[0].get("n")
            if node:
                return dict(node)

        return None

    def get_node_relationships(
        self,
        node_name: str,
        direction: str = "both",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get relationships for a node.

        Args:
            node_name: The name of the node
            direction: Direction of relationships ("in", "out", or "both")
            limit: Maximum number of relationships to return

        Returns:
            List of relationship dictionaries
        """
        if direction == "in":
            cypher_query = """
            MATCH (n:Entity {name: $name})<-[r]-(m:Entity)
            RETURN type(r) as relationship_type, m.name as related_node, m.type as related_type, properties(r) as properties
            LIMIT $limit
            """
        elif direction == "out":
            cypher_query = """
            MATCH (n:Entity {name: $name})-[r]->(m:Entity)
            RETURN type(r) as relationship_type, m.name as related_node, m.type as related_type, properties(r) as properties
            LIMIT $limit
            """
        else:  # both
            cypher_query = """
            MATCH (n:Entity {name: $name})-[r]-(m:Entity)
            RETURN type(r) as relationship_type, m.name as related_node, m.type as related_type, properties(r) as properties
            LIMIT $limit
            """

        results = self.execute_query(cypher_query, {"name": node_name, "limit": limit})

        return results

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        stats = {}

        # Get node count
        cypher_query = "MATCH (n:Entity) RETURN count(n) as node_count"
        results = self.execute_query(cypher_query)
        stats["node_count"] = results[0].get("node_count", 0) if results else 0

        # Get relationship count
        cypher_query = "MATCH ()-[r]->() RETURN count(r) as relationship_count"
        results = self.execute_query(cypher_query)
        stats["relationship_count"] = results[0].get("relationship_count", 0) if results else 0

        # Get node types
        cypher_query = """
        MATCH (n:Entity)
        RETURN n.type as type, count(n) as count
        ORDER BY count DESC
        """
        results = self.execute_query(cypher_query)
        stats["node_types"] = [
            {"type": r.get("type", ""), "count": r.get("count", 0)}
            for r in results
        ]

        # Get relationship types
        cypher_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        results = self.execute_query(cypher_query)
        stats["relationship_types"] = [
            {"type": r.get("type", ""), "count": r.get("count", 0)}
            for r in results
        ]

        return stats
