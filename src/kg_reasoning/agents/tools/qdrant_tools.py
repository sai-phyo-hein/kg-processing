"""Qdrant tools for multi-agent reasoning system.

Provides LangChain tools for querying Qdrant registries to get canonical entities,
predicates, and community metadata.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from kg_extractor.utils.model_setup import (
    OPENAI_EMBEDDING_MODEL,
    EVIDENCE_EMBEDDING_MODEL,
    EVIDENCE_VECTOR_DIM,
    get_embedding_client,
)
from kg_extractor.utils.sparse_vectors import compute_query_sparse
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
        for registry_key in ("entity_registry", "predicate_registry", "metadata_registry", "label_registry", "evidence_registry"):
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
        """Get embeddings for texts using the configured EMBEDDING_PROVIDER."""
        try:
            client = get_embedding_client()
            response = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Warning: Failed to get embeddings: {e}")
            import random
            return [[random.random() for _ in range(1536)] for _ in texts]

    def _get_evidence_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with text-embedding-3-large at EVIDENCE_VECTOR_DIM dimensions.

        Uses a separate model from entity/predicate embeddings because the evidence
        collection stores 3072-dim vectors (vs 1536 for entities/predicates).
        """
        try:
            client = get_embedding_client()
            texts = [t if t.strip() else " " for t in texts]
            response = client.embeddings.create(
                model=EVIDENCE_EMBEDDING_MODEL,
                input=texts,
                dimensions=EVIDENCE_VECTOR_DIM,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Warning: Failed to get evidence embeddings: {e}")
            return [[0.0] * EVIDENCE_VECTOR_DIM for _ in texts]

    def _query_registry_parallel(
        self,
        registry_key: str,
        names: List[str],
        limit: int,
        threshold: float,
        embeddings: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Parallel canonical ID lookup shared by entity and predicate methods.

        Submits all (name, embedding, vector_name) Qdrant queries concurrently,
        then deduplicates per canonical_id keeping the best score.
        """
        spec = self._registry_specs[registry_key]
        collection = spec["collection_name"]
        populated = [v for v in spec["vector_names"] if v.endswith(("_en", "_th"))]

        # Build all query tasks upfront
        tasks: List[tuple] = []  # (name, embedding, vector_name)
        for name, embedding in zip(names, embeddings):
            for vector_name in populated:
                tasks.append((name, embedding, vector_name))

        def _run_query(task: tuple) -> List[tuple]:
            """Execute a single Qdrant query. Returns [(point, query_name)]."""
            name, embedding, vector_name = task
            try:
                search_results = self.client.query_points(
                    collection_name=collection,
                    query=embedding,
                    using=vector_name,
                    limit=limit,
                    with_payload=True,
                )
                return [(point, name) for point in search_results.points if point.score >= threshold]
            except Exception as e:
                print(f"Warning: Failed to query {registry_key} '{name}' via {vector_name}: {e}")
                return []

        # Run all queries concurrently
        all_hits: List[tuple] = []
        max_workers = min(len(tasks), 8) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for hits in pool.map(_run_query, tasks):
                all_hits.extend(hits)

        # Deduplicate by canonical_id per query name, keeping best score
        # Key: (query_name, canonical_id) → (score, point)
        best: Dict[tuple, tuple] = {}
        for point, query_name in all_hits:
            cid = point.payload.get("canonical_id") or str(point.id)
            key = (query_name, cid)
            if key not in best or point.score > best[key][0]:
                best[key] = (point.score, point)

        results: List[Dict[str, Any]] = []
        for (query_name, cid), (score, point) in best.items():
            results.append({
                "query": query_name,
                "canonical_id": cid,
                "name": point.payload.get("name"),
                "score": score,
                "metadata": point.payload,
            })

        return results

    def get_canonical_entities(
        self, entity_names: List[str], limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get canonical entity IDs from entity_registry.

        Queries all (entity × vector_name) combinations concurrently.

        Args:
            entity_names: List of entity names to search for
            limit: Maximum number of results per entity
            threshold: Minimum similarity threshold

        Returns:
            List of canonical entities with IDs and metadata
        """
        if not entity_names:
            return []

        embeddings = self._get_embeddings(entity_names)
        return self._query_registry_parallel(
            registry_key="entity_registry",
            names=entity_names,
            limit=limit,
            threshold=threshold,
            embeddings=embeddings,
        )

    def get_canonical_predicates(
        self, predicate_names: List[str], limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get canonical predicate IDs from predicate_registry.

        Queries all (predicate × vector_name) combinations concurrently.

        Args:
            predicate_names: List of predicate names to search for
            limit: Maximum number of results per predicate
            threshold: Minimum similarity threshold

        Returns:
            List of canonical predicates with IDs and metadata
        """
        if not predicate_names:
            return []

        embeddings = self._get_embeddings(predicate_names)
        return self._query_registry_parallel(
            registry_key="predicate_registry",
            names=predicate_names,
            limit=limit,
            threshold=threshold,
            embeddings=embeddings,
        )

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

    def get_all_community_ids(self) -> List[str]:
        """Get all unique community IDs from metadata_registry.

        Returns:
            List of unique community IDs
        """
        try:
            spec = self._registry_specs["metadata_registry"]
            community_ids = set()
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
                    if community_id:
                        community_ids.add(community_id)

                offset = scroll_result[1]
                if offset is None:
                    break

            return sorted(list(community_ids))

        except Exception as e:
            print(f"Warning: Failed to query all community IDs: {e}")
            return []

    def get_labels_by_group(
        self, label_group: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get labels from label_registry filtered by group.

        Args:
            label_group: The label group to filter by (e.g., "Community", "Person")
            limit: Maximum number of results to return

        Returns:
            List of labels with their metadata
        """
        if not label_group:
            return []

        try:
            spec = self._registry_specs["label_registry"]
            results = []
            offset = None

            while len(results) < limit:
                scroll_result = self.client.scroll(
                    collection_name=spec["collection_name"],
                    scroll_filter={
                        "must": [
                            {
                                "key": "group",
                                "match": {"value": label_group}
                            }
                        ]
                    },
                    limit=min(100, limit - len(results)),
                    offset=offset,
                    with_payload=True,
                )

                for point in scroll_result[0]:
                    results.append({
                        "canonical_id": point.payload.get("canonical_id") or str(point.id),
                        "name": point.payload.get("name"),
                        "group": point.payload.get("group"),
                        "label": point.payload.get("label"),
                        "type": point.payload.get("type"),
                        "metadata": point.payload,
                    })

                offset = scroll_result[1]
                if offset is None:
                    break

            return results

        except Exception as e:
            print(f"Warning: Failed to query labels by group '{label_group}': {e}")
            return []

    def get_entities_by_labels(
        self, label_names: List[str], limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get entities from entity_registry filtered by label.

        Args:
            label_names: List of label names to filter by
            limit: Maximum number of results per label

        Returns:
            List of entities with their metadata
        """
        if not label_names:
            return []

        results = []

        try:
            spec = self._registry_specs["entity_registry"]

            for label_name in label_names:
                offset = None
                label_results = []

                while len(label_results) < limit:
                    scroll_result = self.client.scroll(
                        collection_name=spec["collection_name"],
                        scroll_filter={
                            "must": [
                                {
                                    "key": "label",
                                    "match": {"value": label_name}
                                }
                            ]
                        },
                        limit=min(100, limit - len(label_results)),
                        offset=offset,
                        with_payload=True,
                    )

                    for point in scroll_result[0]:
                        label_results.append({
                            "canonical_id": point.payload.get("canonical_id") or str(point.id),
                            "name": point.payload.get("name"),
                            "label": point.payload.get("label"),
                            "type": point.payload.get("type"),
                            "metadata": point.payload,
                        })

                    offset = scroll_result[1]
                    if offset is None:
                        break

                results.extend(label_results)

        except Exception as e:
            print(f"Warning: Failed to query entities by labels: {e}")

        return results

    @staticmethod
    def _sparse_vector_names(spec: Dict[str, Any]) -> Dict[str, str]:
        """Map dense vector suffix ('_en' or '_th') to its sparse vector name.

        Reads ``sparse_vector_name_en`` / ``sparse_vector_name_th`` from the
        spec.  Returns a dict like ``{"_en": "evidence_vector_en", ...}``.
        """
        mapping: Dict[str, str] = {}
        for suffix in ("_en", "_th"):
            key = f"sparse_vector_name{suffix}"
            name = spec.get(key, "")
            if name:
                mapping[suffix] = name
        return mapping

    def search_evidence(
        self, query_texts: List[str], limit: int = 5, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search evidence registry with hybrid dense + sparse (RRF fusion).

        Language variants (_en, _th) are queried concurrently for each query text.
        Deduplicates across languages by point ID (keeps best score).

        Args:
            query_texts: List of text queries to search for
            limit: Maximum number of results per query
            threshold: Minimum similarity threshold

        Returns:
            List of matching evidence records with entities, predicates, and quotes
        """
        from qdrant_client.http import models as qm

        if not query_texts:
            return []

        spec = self._registry_specs["evidence_registry"]
        collection = spec["collection_name"]
        populated = [v for v in spec["vector_names"] if v.endswith(("_en", "_th"))]
        sparse_map = self._sparse_vector_names(spec)
        embeddings = self._get_evidence_embeddings(query_texts)

        def _search_language(args: tuple):
            """Run one (query_text, embedding, dense_name) search."""
            query_text, embedding, dense_name = args
            suffix = "_en" if dense_name.endswith("_en") else "_th"
            sparse_name = sparse_map.get(suffix, "")

            try:
                # ── Hybrid: dense + sparse → RRF fusion ──────────────
                if sparse_name:
                    query_sparse = compute_query_sparse(query_text)
                    prefetch: list = [
                        qm.Prefetch(
                            query=embedding,
                            using=dense_name,
                            limit=limit * 4,
                        ),
                    ]
                    if query_sparse["indices"]:
                        prefetch.append(
                            qm.Prefetch(
                                query=qm.SparseVector(
                                    indices=query_sparse["indices"],
                                    values=query_sparse["values"],
                                ),
                                using=sparse_name,
                                limit=limit * 4,
                            )
                        )
                    return self.client.query_points(
                        collection_name=collection,
                        prefetch=prefetch,
                        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                        limit=limit,
                        with_payload=True,
                    )
                else:
                    raise ValueError("no sparse vector")
            except Exception:
                # ── Fallback: dense-only ─────────────────────────────
                try:
                    return self.client.query_points(
                        collection_name=collection,
                        query=embedding,
                        using=dense_name,
                        limit=limit,
                        with_payload=True,
                    )
                except Exception as e:
                    print(f"Warning: Evidence search failed for '{query_text}': {e}")
                    return None

        # Build all (query_text, embedding, dense_name) tasks
        tasks = []
        for query_text, embedding in zip(query_texts, embeddings):
            for dense_name in populated:
                tasks.append((query_text, embedding, dense_name))

        # Run all language searches concurrently
        all_results: List[tuple] = []  # (query_text, search_results)
        max_workers = min(len(tasks), 8) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for task, result in zip(tasks, pool.map(_search_language, tasks)):
                if result is not None:
                    all_results.append((task[0], result))

        # Deduplicate across languages per query text
        results_map: Dict[str, Dict[str, Any]] = {}  # query_text -> {seen, points_by_id}
        for query_text, search_results in all_results:
            if query_text not in results_map:
                results_map[query_text] = {"seen": {}, "points_by_id": {}}
            entry = results_map[query_text]

            for point in search_results.points:
                if point.score >= threshold:
                    pid = str(point.id)
                    if pid not in entry["seen"] or point.score > entry["seen"][pid]:
                        entry["seen"][pid] = point.score
                        entry["points_by_id"][pid] = point

        results = []
        for query_text, entry in results_map.items():
            for point in entry["points_by_id"].values():
                results.append({
                    "query": query_text,
                    "score": point.score,
                    "entities": point.payload.get("entities", []),
                    "predicates": point.payload.get("predicates", []),
                    "evidence_quote": point.payload.get("evidence_quote", ""),
                    "evidence_quote_en": point.payload.get("evidence_quote_en", ""),
                    "community_id": point.payload.get("community_id"),
                })

        return results


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


@tool
def get_labels_by_group(label_group: str, limit: int = 50) -> str:
    """Get labels from Qdrant label_registry filtered by label group.

    Use this tool when the user query relates to a specific domain/category.
    First identify the label_group from configs/label_group_config.yaml, then use
    this tool to retrieve all labels in that group.

    Payload schema:
    - canonical_id (uuid): Unique identifier for the label
    - name (keyword): Label name
    - group (keyword): Label group (e.g., "Community", "Person", "Activity")
    - label (keyword): Label type
    - type (keyword): Entity type

    Args:
        label_group: The label group to filter by (e.g., "Community", "Person", "Activity")
        limit: Maximum number of results to return (default: 50)

    Returns:
        JSON string with labels matching the specified group
    """
    manager = _get_manager()
    results = manager.get_labels_by_group(label_group, limit)
    return json.dumps(results, indent=2)


@tool
def get_entities_by_labels(label_names: str, limit: int = 50) -> str:
    """Get entities from Qdrant entity_registry filtered by label.

    Use this tool after identifying relevant labels to find entities of those types.
    This helps narrow down entities based on their classification.

    Payload schema:
    - canonical_id (uuid): Unique identifier for the entity
    - name (keyword): Entity name
    - label (keyword): Entity label/type
    - type (keyword): Entity type

    Args:
        label_names: Comma-separated list of label names to filter by
        limit: Maximum number of results per label (default: 50)

    Returns:
        JSON string with entities matching the specified labels
    """
    manager = _get_manager()
    names = [name.strip() for name in label_names.split(",") if name.strip()]
    results = manager.get_entities_by_labels(names, limit)
    return json.dumps(results, indent=2)


@tool
def get_all_community_ids() -> str:
    """Get all unique community IDs from Qdrant metadata_registry.

    Use this tool to retrieve the complete list of available community IDs in the knowledge graph.
    This is useful for understanding what communities exist and for filtering queries by community.

    Returns:
        JSON string with list of all unique community IDs (e.g., ["หมู่ 1_village", "หมู่ 2_village"])
    """
    manager = _get_manager()
    results = manager.get_all_community_ids()
    return json.dumps(results, indent=2)


@tool
def search_evidence(query_texts: str, limit: int = 5, threshold: float = 0.7) -> str:
    """Search evidence registry for evidence quotes matching the query.

    Use this tool to find source evidence quotes and their associated entities
    and predicates. Useful for answering questions about what evidence supports
    particular claims or relationships.

    Payload schema:
    - evidence_quote (text): Original source evidence text
    - evidence_quote_en (text): English translation of evidence
    - entities (keyword[]): List of entity names mentioned in evidence
    - predicates (keyword[]): List of predicate names in evidence
    - community_id (keyword): Community identifier

    Args:
        query_texts: Comma-separated list of query texts to search for evidence
        limit: Maximum number of results per query (default: 5)
        threshold: Minimum similarity threshold 0-1 (default: 0.7)

    Returns:
        JSON string with matching evidence records
    """
    manager = _get_manager()
    queries = [q.strip() for q in query_texts.split(",") if q.strip()]
    results = manager.search_evidence(queries, limit, threshold)
    return json.dumps(results, indent=2)
