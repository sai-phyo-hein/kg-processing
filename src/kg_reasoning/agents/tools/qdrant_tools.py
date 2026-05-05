"""Qdrant tools for multi-agent reasoning system.

Provides LangChain tools for querying Qdrant registries to get canonical entities,
predicates, and community metadata.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from qdrant_client import QdrantClient

load_dotenv()


class QdrantToolsManager:
    """Manager for Qdrant tools with shared client and configuration."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        """Initialize Qdrant tools manager.

        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        # Initialize client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        # Load registry specs from JSON files
        project_root = Path(__file__).parent.parent.parent.parent.parent
        self.registry_info_dir = project_root / "registry_info"
        self._registry_specs = self._load_registry_specs()

    def _load_registry_specs(self) -> Dict[str, Dict[str, Any]]:
        """Load collection names and vector names from registry_info JSON files.

        Returns:
            Dict mapping registry key to its spec (collection_name, vector_names).
        """
        specs = {}
        for registry_key in ("entity_registry", "predicate_registry", "metadata_registry"):
            spec_file = self.registry_info_dir / f"{registry_key}.json"
            with open(spec_file) as f:
                data = json.load(f)
            vector_names = [
                data[key]
                for key in ("vector_name", "vector_name_en", "vector_name_th")
                if data.get(key)
            ]
            specs[registry_key] = {
                "collection_name": data["collection_name"],
                "vector_names": vector_names,
            }
        return specs

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using OpenAI.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Warning: Failed to get embeddings: {e}")
            import random
            return [[random.random() for _ in range(1536)] for _ in texts]

    def get_canonical_entities(
        self, entity_names: List[str], limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get canonical entity IDs from entity_registry.

        Args:
            entity_names: List of entity names to search for
            limit: Maximum number of results per entity
            threshold: Minimum similarity threshold

        Returns:
            List of canonical entities with IDs and metadata
        """
        if not entity_names:
            return []

        results = []

        # Get embeddings for all entities
        embeddings = self._get_embeddings(entity_names)

        for entity_name, embedding in zip(entity_names, embeddings):
            try:
                spec = self._registry_specs["entity_registry"]
                seen: Dict[str, float] = {}  # canonical_id -> best score
                points_by_id: Dict[str, Any] = {}

                for vector_name in spec["vector_names"]:
                    search_results = self.client.query_points(
                        collection_name=spec["collection_name"],
                        query=embedding,
                        using=vector_name,
                        limit=limit,
                        with_payload=True,
                    )
                    for point in search_results.points:
                        if point.score >= threshold:
                            cid = point.payload.get("canonical_id") or str(point.id)
                            if cid not in seen or point.score > seen[cid]:
                                seen[cid] = point.score
                                points_by_id[cid] = point

                for point in points_by_id.values():
                    results.append({
                        "query": entity_name,
                        "canonical_id": point.payload.get("canonical_id") or str(point.id),
                        "name": point.payload.get("name"),
                        "score": point.score,
                        "metadata": point.payload,
                    })
            except Exception as e:
                print(f"Warning: Failed to query entity '{entity_name}': {e}")

        return results

    def get_canonical_predicates(
        self, predicate_names: List[str], limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get canonical predicate IDs from predicate_registry.

        Args:
            predicate_names: List of predicate names to search for
            limit: Maximum number of results per predicate
            threshold: Minimum similarity threshold

        Returns:
            List of canonical predicates with IDs and metadata
        """
        if not predicate_names:
            return []

        results = []

        # Get embeddings for all predicates
        embeddings = self._get_embeddings(predicate_names)

        for predicate_name, embedding in zip(predicate_names, embeddings):
            try:
                spec = self._registry_specs["predicate_registry"]
                seen: Dict[str, float] = {}  # canonical_id -> best score
                points_by_id: Dict[str, Any] = {}

                for vector_name in spec["vector_names"]:
                    search_results = self.client.query_points(
                        collection_name=spec["collection_name"],
                        query=embedding,
                        using=vector_name,
                        limit=limit,
                        with_payload=True,
                    )
                    for point in search_results.points:
                        if point.score >= threshold:
                            cid = point.payload.get("canonical_id") or str(point.id)
                            if cid not in seen or point.score > seen[cid]:
                                seen[cid] = point.score
                                points_by_id[cid] = point

                for point in points_by_id.values():
                    results.append({
                        "query": predicate_name,
                        "canonical_id": point.payload.get("canonical_id") or str(point.id),
                        "name": point.payload.get("name"),
                        "score": point.score,
                        "metadata": point.payload,
                    })
            except Exception as e:
                print(f"Warning: Failed to query predicate '{predicate_name}': {e}")

        return results

    def get_community_metadata(
        self, community_ids: Optional[List[str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get community metadata from metadata_registry.

        Args:
            community_ids: Optional list of specific community IDs to retrieve
            limit: Maximum number of results to return

        Returns:
            List of community metadata entries
        """
        try:
            spec = self._registry_specs["metadata_registry"]
            if community_ids:
                results = []
                offset = None

                while True:
                    scroll_result = self.client.scroll(
                        collection_name=spec["collection_name"],
                        limit=100,
                        offset=offset,
                        with_payload=True,
                    )

                    for point in scroll_result[0]:
                        community_id = point.payload.get("community_id")
                        if community_id in community_ids:
                            results.append({
                                "community_id": community_id,
                                "metadata": point.payload,
                            })

                        if len(results) >= limit:
                            return results

                    offset = scroll_result[1]
                    if offset is None:
                        break

                return results
            else:
                results = []
                offset = None

                while len(results) < limit:
                    scroll_result = self.client.scroll(
                        collection_name=spec["collection_name"],
                        limit=min(100, limit - len(results)),
                        offset=offset,
                        with_payload=True,
                    )

                    for point in scroll_result[0]:
                        results.append({
                            "community_id": point.payload.get("community_id"),
                            "metadata": point.payload,
                        })

                    offset = scroll_result[1]
                    if offset is None:
                        break

                return results

        except Exception as e:
            print(f"Warning: Failed to query metadata registry: {e}")
            return []


# Create global instance for tools
_manager = None


def _get_manager() -> QdrantToolsManager:
    """Get or create global QdrantToolsManager instance."""
    global _manager
    if _manager is None:
        _manager = QdrantToolsManager()
    return _manager


@tool
def get_canonical_entities(entity_names: str, limit: int = 5, threshold: float = 0.7) -> str:
    """Get canonical entity IDs from Qdrant entity_registry.

    Use this tool to find canonical IDs and names for entities mentioned in the user query.
    This helps standardize entity references for graph queries.

    Payload schema:
    - canonical_id (uuid): Unique identifier for the canonical entity
    - name (keyword): Human-readable canonical name of the entity

    Args:
        entity_names: Comma-separated list of entity names to search for
        limit: Maximum number of results per entity (default: 5)
        threshold: Minimum similarity threshold 0-1 (default: 0.7)

    Returns:
        JSON string with canonical entity matches. Each result contains:
        "canonical_id" (uuid) and "name" (keyword) from the payload, plus "score".
    """
    manager = _get_manager()
    names = [name.strip() for name in entity_names.split(",") if name.strip()]
    results = manager.get_canonical_entities(names, limit, threshold)
    return json.dumps(results, indent=2)


@tool
def get_canonical_predicates(
    predicate_names: str, limit: int = 5, threshold: float = 0.7
) -> str:
    """Get canonical predicate IDs from Qdrant predicate_registry.

    Use this tool to find canonical IDs and names for relationships/predicates in the query.
    This helps standardize relationship references for graph queries.

    Payload schema:
    - canonical_id (uuid): Unique identifier for the canonical predicate
    - name (keyword): Human-readable canonical name of the predicate

    Args:
        predicate_names: Comma-separated list of predicate names to search for
        limit: Maximum number of results per predicate (default: 5)
        threshold: Minimum similarity threshold 0-1 (default: 0.7)

    Returns:
        JSON string with canonical predicate matches. Each result contains:
        "canonical_id" (uuid) and "name" (keyword) from the payload, plus "score".
    """
    manager = _get_manager()
    names = [name.strip() for name in predicate_names.split(",") if name.strip()]
    results = manager.get_canonical_predicates(names, limit, threshold)
    return json.dumps(results, indent=2)


@tool
def get_community_metadata(community_ids: str = "", limit: int = 100) -> str:
    """Get community metadata from Qdrant metadata_registry.

    Use this tool to retrieve information about knowledge graph communities.
    Communities group related entities and can be used to filter graph queries.

    Args:
        community_ids: Comma-separated list of community IDs (empty for all communities)
        limit: Maximum number of results to return (default: 100)

    Returns:
        JSON string with community metadata entries
    """
    manager = _get_manager()
    ids = [cid.strip() for cid in community_ids.split(",") if cid.strip()] if community_ids else None
    results = manager.get_community_metadata(ids, limit)
    return json.dumps(results, indent=2)
