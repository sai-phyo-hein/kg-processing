"""Qdrant matching module for entity resolution."""

import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class QdrantMatcher:
    """Match entities in Qdrant using vector similarity."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        registry_info_dir: Optional[str] = None,
    ):
        """Initialize the Qdrant matcher.

        Args:
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            registry_info_dir: Directory containing registry specification files
        """
        import os
        from pathlib import Path

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        # Set registry info directory
        if registry_info_dir is None:
            # Default to registry_info in the project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.registry_info_dir = project_root / "registry_info"
        else:
            self.registry_info_dir = Path(registry_info_dir)

        # Load registry specifications
        self.registry_specs = self._load_registry_specs()

        # Initialize Qdrant client
        self._init_qdrant_client()

    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient

            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}")

    def _load_registry_specs(self) -> Dict[str, Dict[str, Any]]:
        """Load registry specifications from JSON files.

        Returns:
            Dictionary mapping collection names to their specifications
        """
        specs = {}
        collections = ["entity_registry", "predicate_registry", "ontology_registry"]

        for collection_name in collections:
            spec_file = self.registry_info_dir / f"{collection_name}.json"
            if spec_file.exists():
                with open(spec_file, "r") as f:
                    specs[collection_name] = json.load(f)
            else:
                print(f"Warning: Specification file not found: {spec_file}")
                # Use default specification
                specs[collection_name] = {
                    "collection_name": collection_name,
                    "vector_name": "entity",  # Default fallback
                    "vector_config": {
                        "size": 1536,
                        "distance": "Cosine"
                    }
                }

        return specs

    def _get_vector_name(self, collection_name: str) -> str:
        """Get the vector name for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Vector name for the collection
        """
        spec = self.registry_specs.get(collection_name, {})
        return spec.get("vector_name", "entity")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            import os
            from openai import OpenAI

            openai_api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=openai_api_key)

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Failed to get embedding: {e}")
            # Fallback to random vector
            import random
            return [random.random() for _ in range(1536)]

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            import os
            from openai import OpenAI

            openai_api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=openai_api_key)

            # Process in batches of 100 (OpenAI limit)
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            print(f"Warning: Failed to get batch embeddings: {e}")
            # Fallback to random vectors
            import random
            return [[random.random() for _ in range(1536)] for _ in texts]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        try:
            import numpy as np
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception:
            # Fallback to manual calculation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)

    def _query_qdrant_with_embedding(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query Qdrant for similar entities using existing embedding.

        Args:
            collection_name: Name of the collection
            query_vector: Pre-computed embedding vector
            limit: Maximum number of results to return

        Returns:
            List of matching points with payloads
        """
        try:
            from qdrant_client.http import models as qdrant_models

            # Get vector name for this collection
            vector_name = self._get_vector_name(collection_name)

            # Search using vector similarity with query_points
            search_results = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
            )

            # Convert to expected format
            matches = []
            for result in search_results.points:
                matches.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })

            return matches

        except Exception as e:
            print(f"Warning: Failed to query Qdrant: {e}")
            return []

    def match_entities(
        self,
        entities: List[Dict[str, Any]],
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """Match entities in Qdrant using vector similarity with batch processing.

        Args:
            entities: List of entity dictionaries with name and type
            similarity_threshold: Minimum similarity score for a match

        Returns:
            List of matched entities with canonical forms
        """
        if not entities:
            return []

        matches = []

        # Get embeddings for all entities
        entity_names = [entity["name"] for entity in entities]
        embeddings = self._get_embeddings_batch(entity_names)

        # Process entities using similarity-based grouping
        processed_canonicals = set()  # Track which canonicals we've already processed

        for i, (entity, embedding) in enumerate(zip(entities, embeddings)):
            # Skip if we've already processed this entity as part of a group
            if entity["name"] in processed_canonicals:
                continue

            # Determine collection based on entity type
            if entity["type"] == "predicate":
                collection_name = "predicate_registry"
            elif entity["type"] in ["ontology", "type"]:
                collection_name = "ontology_registry"
            else:
                collection_name = "entity_registry"

            # Query Qdrant for similar entities (will find previously processed ones)
            qdrant_matches = self._query_qdrant_with_embedding(
                collection_name,
                embedding,
                limit=5,
            )

            # Find similar entities in current batch using cosine similarity
            similar_entities = []
            for j, (other_entity, other_embedding) in enumerate(zip(entities, embeddings)):
                if i == j:
                    continue
                # Calculate cosine similarity
                similarity = self._cosine_similarity(embedding, other_embedding)
                if similarity > similarity_threshold:
                    similar_entities.append({
                        "id": None,  # No ID yet, not in Qdrant
                        "score": similarity,
                        "payload": {
                            "name": other_entity["name"],
                            "type": other_entity["type"],
                        },
                    })

            # Combine Qdrant matches with similar entities from current batch
            all_matches = qdrant_matches + similar_entities

            # Find best match above threshold
            best_match = None
            best_score = 0.0

            for match in all_matches:
                score = match["score"]
                if score >= similarity_threshold and score > best_score:
                    best_match = match
                    best_score = score

            if best_match:
                canonical_name = best_match["payload"].get("name", entity["name"])
                canonical_id = best_match.get("id")

                matches.append({
                    "original": entity["name"],
                    "canonical": canonical_name,
                    "canonical_id": canonical_id,
                    "score": best_score,
                    "type": entity["type"],
                    "collection": collection_name,
                })

                # Mark this entity as processed
                processed_canonicals.add(entity["name"])

                # Mark similar entities as processed too (they map to the same canonical)
                for similar_entity in similar_entities:
                    similar_name = similar_entity["payload"]["name"]
                    if similar_name not in processed_canonicals:
                        matches.append({
                            "original": similar_name,
                            "canonical": canonical_name,
                            "canonical_id": canonical_id,
                            "score": similar_entity["score"],
                            "type": similar_entity["payload"]["type"],
                            "collection": collection_name,
                        })
                        processed_canonicals.add(similar_name)

        return matches

    def has_matches(
        self,
        entities: List[Dict[str, Any]],
        similarity_threshold: float = 0.75,
    ) -> bool:
        """Check if any entities have matches in Qdrant.

        Args:
            entities: List of entity dictionaries
            similarity_threshold: Minimum similarity score for a match

        Returns:
            True if any entity has a match, False otherwise
        """
        matches = self.match_entities(entities, similarity_threshold)
        return len(matches) > 0
