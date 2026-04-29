"""Neo4j integration module for knowledge graph storage."""

import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.schema_parser import get_schema_parser


class Neo4jGraphBuilder:
    """Build knowledge graphs in Neo4j from refined triples."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        with_schema: bool = False,
        schema_path: Optional[str] = None,
    ):
        """Initialize the Neo4j graph builder.

        Args:
            neo4j_uri: Neo4j server URI (from env if not provided)
            neo4j_user: Neo4j username (from env if not provided)
            neo4j_password: Neo4j password (from env if not provided)
            with_schema: Whether to validate against schema (default: False)
            schema_path: Path to schema.md file (default: utils/schema.md)
        """
        import os

        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.with_schema = with_schema

        # Initialize schema parser if with_schema is enabled
        self.schema_parser = None
        if self.with_schema:
            self.schema_parser = get_schema_parser(schema_path)

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

    def _create_constraints(self):
        """Create constraints for unique nodes."""
        constraints = [
            "CREATE CONSTRAINT qdrant_id_constraint IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print("Index/Constraint created successfully.")
                except Exception as e:
                    print(f"Warning: Failed to create constraint: {e}")

    def _validate_entity_type(self, entity_type: str) -> bool:
        """Validate entity type against schema.

        Args:
            entity_type: Entity type to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.with_schema or self.schema_parser is None:
            return True

        # Map common entity types to schema node types
        type_mapping = {
            "Entity": "Entity",  # Generic entity
            "SocialCapital": "SocialCapital",
            "Activity": "Activity",
            "Impact": "Impact",
            "TargetGroup": "TargetGroup",
            "Domain": "Domain",
            "Tambon": "Tambon",
            "Village": "Village",
            "Evidence": "Evidence",
        }

        # Check if the type is in our mapping or directly valid
        mapped_type = type_mapping.get(entity_type, entity_type)
        return self.schema_parser.validate_node_type(mapped_type)

    def _validate_relation(
        self,
        subject_type: str,
        relation: str,
        object_type: str,
    ) -> bool:
        """Validate relation against schema.

        Args:
            subject_type: Type of subject entity
            relation: Relation name
            object_type: Type of object entity

        Returns:
            True if valid, False otherwise
        """
        if not self.with_schema or self.schema_parser is None:
            return True

        # Map common entity types to schema node types
        type_mapping = {
            "Entity": "Entity",  # Generic entity
            "SocialCapital": "SocialCapital",
            "Activity": "Activity",
            "Impact": "Impact",
            "TargetGroup": "TargetGroup",
            "Domain": "Domain",
            "Tambon": "Tambon",
            "Village": "Village",
            "Evidence": "Evidence",
        }

        mapped_subject = type_mapping.get(subject_type, subject_type)
        mapped_object = type_mapping.get(object_type, object_type)

        return self.schema_parser.validate_relation(mapped_subject, relation, mapped_object)

    def _batch_create_nodes(
        self,
        nodes_data: List[Dict[str, Any]],
        node_type: str,
    ) -> int:
        """Batch create nodes using UNWIND.

        Args:
            nodes_data: List of node data dictionaries
            node_type: Type of node (Entity, Predicate, Ontology)

        Returns:
            Number of nodes created
        """
        if not nodes_data:
            return 0

        # Validate node type if schema validation is enabled
        if not self._validate_entity_type(node_type):
            print(f"Warning: Invalid node type '{node_type}' according to schema. Skipping node creation.")
            return 0

        with self.driver.session() as session:
            query = f"""
            UNWIND $nodes_data AS node
            MERGE (n:{node_type} {{canonical_id: node.canonical_id}})
            SET n.name = node.name,
                n.type = node.type,
                n.updated_at = datetime()
            RETURN count(n) as count
            """

            result = session.run(query, nodes_data=nodes_data)
            record = result.single()
            return record["count"] if record else 0

    def _batch_create_relationships(
        self,
        relationships_data: List[Dict[str, Any]],
    ) -> int:
        """Batch create relationships using UNWIND and APOC.

        Args:
            relationships_data: List of relationship data dictionaries

        Returns:
            Number of relationships created
        """
        if not relationships_data:
            return 0

        # Filter relationships based on schema validation
        valid_relationships = []
        for rel in relationships_data:
            # Get source and target types from the relationship data
            # We need to look up the entity types from the nodes
            source_type = rel.get("source_type", "Entity")
            target_type = rel.get("target_type", "Entity")
            relation = rel.get("edge_type", "")

            # Validate relation if schema validation is enabled
            if self._validate_relation(source_type, relation, target_type):
                valid_relationships.append(rel)
            else:
                print(
                    f"Warning: Invalid relation '{relation}' from {source_type} to {target_type} "
                    f"according to schema. Skipping relationship creation."
                )

        if not valid_relationships:
            return 0

        with self.driver.session() as session:
            query = """
            UNWIND $relationships_data AS rel
            MATCH (source:Entity {canonical_id: rel.source_id})
            MATCH (target:Entity {canonical_id: rel.target_id})
            CALL apoc.merge.relationship(
                source,
                rel.edge_type,
                {},
                rel.edge_props,
                target
            ) YIELD rel as relationship
            RETURN count(relationship) as count
            """

            result = session.run(query, relationships_data=valid_relationships)
            record = result.single()
            return record["count"] if record else 0

    def build_graph_from_triples(
        self,
        refined_triples_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build Neo4j graph from refined triples using batch processing.

        Args:
            refined_triples_data: Refined triples data

        Returns:
            Statistics about the graph building process
        """
        # Create constraints
        self._create_constraints()

        stats = {
            "entities_created": 0,
            "relationships_created": 0,
            "errors": [],
        }

        # Track created nodes to avoid duplicates
        created_entities = {}
        relationships_data = []

        for chunk_data in refined_triples_data.get("chunks", []):
            chunk_id = chunk_data["chunk_id"]

            for triple in chunk_data.get("triples", []):
                try:
                    # Get source entity (subject) with canonical_id
                    source_refinement = triple["refinement"]["subject"]
                    source_canonical_id = source_refinement["canonical_id"]
                    source_name = triple["subject"]["name"]
                    source_type = triple["subject"]["type"]

                    if source_canonical_id and source_canonical_id not in created_entities:
                        created_entities[source_canonical_id] = {
                            "canonical_id": source_canonical_id,
                            "name": source_name,
                            "type": source_type,
                        }

                    # Get target entity (object) with canonical_id
                    target_refinement = triple["refinement"]["object"]
                    target_canonical_id = target_refinement["canonical_id"]
                    target_name = triple["object"]["name"]
                    target_type = triple["object"]["type"]

                    if target_canonical_id and target_canonical_id not in created_entities:
                        created_entities[target_canonical_id] = {
                            "canonical_id": target_canonical_id,
                            "name": target_name,
                            "type": target_type,
                        }

                    # Get predicate name as edge type
                    predicate_name = triple["predicate"]

                    # Flatten triple properties for edge properties
                    edge_props = {}
                    for key, value in triple["properties"].items():
                        if key == "causal_link" and isinstance(value, dict):
                            # Flatten causal_link properties
                            for causal_key, causal_value in value.items():
                                edge_props[causal_key] = causal_value
                        else:
                            edge_props[key] = value

                    # Add metadata
                    edge_props["chunk_id"] = chunk_id
                    edge_props["updated_at"] = "datetime()"

                    # Collect relationship data with all triple properties
                    if source_canonical_id and target_canonical_id:
                        relationships_data.append({
                            "source_id": source_canonical_id,
                            "source_type": source_type,
                            "target_id": target_canonical_id,
                            "target_type": target_type,
                            "edge_type": predicate_name,
                            "edge_props": edge_props,
                        })

                except Exception as e:
                    error_msg = f"Chunk {chunk_id}, Triple: {e}"
                    stats["errors"].append(error_msg)
                    print(f"Warning: {error_msg}")

        # Batch create all entity nodes
        if created_entities:
            stats["entities_created"] = self._batch_create_nodes(
                list(created_entities.values()),
                "Entity",
            )

        # Batch create all relationships
        if relationships_data:
            stats["relationships_created"] = self._batch_create_relationships(
                relationships_data,
            )

        return stats


def build_graph_from_file(
    input_path: str,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    with_schema: bool = False,
    schema_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build Neo4j graph from refined triples file.

    Args:
        input_path: Path to refined triples JSON file
        neo4j_uri: Neo4j server URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        with_schema: Whether to validate against schema (default: False)
        schema_path: Path to schema.md file (default: utils/schema.md)

    Returns:
        Statistics about the graph building process
    """
    # Load refined triples
    with open(input_path, "r", encoding="utf-8") as f:
        refined_triples_data = json.load(f)

    # Create graph builder
    builder = Neo4jGraphBuilder(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        with_schema=with_schema,
        schema_path=schema_path,
    )

    try:
        # Build graph
        stats = builder.build_graph_from_triples(refined_triples_data)
        return stats
    finally:
        builder.close()
