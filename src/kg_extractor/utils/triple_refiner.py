"""Triple refinement module for entity resolution using Qdrant."""

import json
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TripleRefiner:
    """Refine knowledge graph triples using Qdrant for entity resolution."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        registry_info_dir: Optional[str] = None,
    ):
        """Initialize the triple refiner.

        Args:
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            llm_provider: LLM provider for canonical comparison
            llm_model: Model for LLM analysis
            registry_info_dir: Directory containing registry specification files
        """
        import os
        from pathlib import Path

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Set registry info directory
        if registry_info_dir is None:
            # Default to registry_info in the project root
            project_root = Path(__file__).parent.parent.parent.parent
            self.registry_info_dir = project_root / "registry_info"
        else:
            self.registry_info_dir = Path(registry_info_dir)

        # Load registry specifications
        self.registry_specs = self._load_registry_specs()

        print(f"Registry info directory: {self.registry_info_dir}")
        print(f"Registry specs loaded: {list(self.registry_specs.keys())}")

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

    def _get_vector_config(self, collection_name: str) -> Dict[str, Any]:
        """Get the vector configuration for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Vector configuration for the collection
        """
        spec = self.registry_specs.get(collection_name, {})
        return spec.get("vector_config", {"size": 1536, "distance": "Cosine"})

    def _ensure_collections_exist(self):
        """Ensure all required collections exist in Qdrant."""
        from qdrant_client.http import models as qdrant_models

        collections = {
            "entity_registry": "Entity registry for subject and object entities",
            "predicate_registry": "Predicate registry for relationship predicates",
            "ontology_registry": "Ontology registry for entity types",
        }

        for collection_name, description in collections.items():
            try:
                # Check if collection exists
                self.qdrant_client.get_collection(collection_name)
                print(f"Collection exists: {collection_name}")
            except Exception:
                # Create collection if it doesn't exist with correct vector configuration
                vector_name = self._get_vector_name(collection_name)
                vector_config = self._get_vector_config(collection_name)

                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: qdrant_models.VectorParams(
                            size=vector_config.get("size", 1536),
                            distance=qdrant_models.Distance.COSINE,
                        )
                    },
                )
                print(f"Created collection: {collection_name}")

    def _generate_uuid(self, name: str) -> str:
        """Generate a deterministic UUID from a name.

        Args:
            name: Name to generate UUID from

        Returns:
            UUID string
        """
        # Create a namespace UUID based on the name
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        return str(uuid.uuid5(namespace, name))

    def _query_qdrant(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query Qdrant for similar entities using vector similarity.

        Args:
            collection_name: Name of the collection to query
            query_text: Text to search for
            limit: Maximum number of results to return

        Returns:
            List of matching points with payloads
        """
        try:
            from qdrant_client.http import models as qdrant_models

            # Get embedding for query text
            query_vector = self._get_embedding(query_text)

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

    def _query_qdrant_with_embedding(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Query Qdrant for similar entities using existing embedding.

        Args:
            collection_name: Name of the collection to query
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

    def _determine_canonical_with_llm(
        self,
        current_entity: str,
        existing_entities: List[Dict[str, Any]],
        entity_type: str,
    ) -> Dict[str, Any]:
        """Use LLM to determine which entity should be canonical.

        Args:
            current_entity: Current entity name
            existing_entities: List of existing entities from Qdrant
            entity_type: Type of entity (subject, predicate, ontology)

        Returns:
            Dictionary with canonical entity and synonyms
        """
        if not existing_entities:
            # No existing entities, current one is canonical
            return {
                "canonical": current_entity,
                "synonyms": [],
                "is_new": True,
            }

        # Create prompt for LLM
        prompt = self._create_canonical_comparison_prompt(
            current_entity,
            existing_entities,
            entity_type,
        )

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response
        return self._parse_canonical_response(response, current_entity, existing_entities)

    def _create_canonical_comparison_prompt(
        self,
        current_entity: str,
        existing_entities: List[Dict[str, Any]],
        entity_type: str,
    ) -> str:
        """Create prompt for canonical comparison.

        Args:
            current_entity: Current entity name
            existing_entities: List of existing entities
            entity_type: Type of entity

        Returns:
            Prompt string
        """
        existing_list = "\n".join([
            f"- {e['payload']['name']} (ID: {e['id']}, Score: {e['score']:.2f})"
            for e in existing_entities
        ])

        prompt = f"""You are an expert in entity resolution and ontology management.

Your task is to determine which entity should be the canonical form among similar entities.

**Entity Type:** {entity_type}

**Current Entity:**
{current_entity}

**Existing Similar Entities:**
{existing_list}

**Instructions:**
1. Analyze the current entity and existing entities
2. Determine which one should be the canonical form (the most standard, complete, and widely used representation)
3. Identify which entities are synonyms or variations of the canonical
4. If the current entity is better than all existing ones, it should become the new canonical

**Output Format (JSON):**
```json
{{
  "canonical": "canonical_entity_name",
  "canonical_id": "existing_id_or_null",
  "synonyms": ["synonym1", "synonym2"],
  "reasoning": "brief explanation of the decision"
}}
```

Analyze the entities and return the canonical form in JSON format:"""

        return prompt

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        try:
            import os
            from langchain_openai import ChatOpenAI
            from kg_extractor.utils.parser import (
                get_api_key,
                get_groq_api_key,
                get_openrouter_api_key,
            )

            # Create LLM based on provider
            if self.llm_provider == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=openai_api_key,
                )
            elif self.llm_provider == "groq":
                groq_api_key = get_groq_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            elif self.llm_provider == "nvidia":
                nvidia_api_key = get_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=nvidia_api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                )
            else:  # openrouter
                openrouter_api_key = get_openrouter_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                )

            # Get response
            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"canonical": null, "canonical_id": null, "synonyms": [], "reasoning": "LLM error"}'

    def _parse_canonical_response(
        self,
        response: str,
        current_entity: str,
        existing_entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Parse LLM response for canonical comparison.

        Args:
            response: LLM response text
            current_entity: Current entity name
            existing_entities: List of existing entities

        Returns:
            Dictionary with canonical entity and synonyms
        """
        try:
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            data = json.loads(response)

            canonical = data.get("canonical")
            canonical_id = data.get("canonical_id")
            synonyms = data.get("synonyms", [])

            # If no canonical determined, use current entity
            if not canonical:
                canonical = current_entity
                canonical_id = None
                synonyms = [e["payload"]["name"] for e in existing_entities]

            return {
                "canonical": canonical,
                "canonical_id": canonical_id,
                "synonyms": synonyms,
                "is_new": canonical_id is None,
            }

        except json.JSONDecodeError:
            # Fallback: current entity is canonical
            return {
                "canonical": current_entity,
                "canonical_id": None,
                "synonyms": [e["payload"]["name"] for e in existing_entities],
                "is_new": True,
            }
        except KeyError as e:
            print(f"Warning: Missing key in LLM response: {e}")
            # Fallback: current entity is canonical
            return {
                "canonical": current_entity,
                "canonical_id": None,
                "synonyms": [],
                "is_new": True,
            }
        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return {
                "canonical": current_entity,
                "canonical_id": None,
                "synonyms": [],
                "is_new": True,
            }

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

    def _batch_upsert_entities(
        self,
        entities: List[Dict[str, Any]],
        collection_name: str,
    ) -> Dict[str, str]:
        """Batch upsert entities to Qdrant.

        Args:
            entities: List of entity dictionaries with name, type, synonyms, source_chunk
            collection_name: Name of the collection

        Returns:
            Dictionary mapping entity names to point IDs
        """
        if not entities:
            return {}

        try:
            from qdrant_client.http import models as qdrant_models

            # Get vector configuration for this collection
            vector_config = self._get_vector_config(collection_name)
            vector_name = self._get_vector_name(collection_name)

            # Get embeddings for all entities in batch
            entity_names = [entity["name"] for entity in entities]
            embeddings = self._get_embeddings_batch(entity_names)

            # Create points
            points = []
            for entity, embedding in zip(entities, embeddings):
                point_id = self._generate_uuid(entity["name"])
                point = qdrant_models.PointStruct(
                    id=point_id,
                    vector={
                        vector_name: embedding
                    },
                    payload={
                        "name": entity["name"],
                        "type": entity["type"],
                        "is_canonical": entity.get("is_canonical", True),
                        "synonyms": entity.get("synonyms", []),
                        "source_chunk": entity["source_chunk"],
                    },
                )
                points.append(point)

            # Batch upsert
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )

            # Return mapping of names to IDs
            return {entity["name"]: self._generate_uuid(entity["name"]) for entity in entities}

        except Exception as e:
            print(f"Warning: Failed to batch upsert entities: {e}")
            return {}

    def _upsert_entity(
        self,
        collection_name: str,
        name: str,
        entity_type: str,
        is_canonical: bool,
        synonyms: List[str],
        source_chunk: int,
    ) -> str:
        """Upsert a single entity to Qdrant (for backward compatibility).

        Args:
            collection_name: Name of the collection
            name: Entity name
            entity_type: Type of entity
            is_canonical: Whether this is the canonical form
            synonyms: List of synonyms
            source_chunk: Source chunk ID

        Returns:
            ID of the upserted point
        """
        result = self._batch_upsert_entities(
            [{
                "name": name,
                "type": entity_type,
                "is_canonical": is_canonical,
                "synonyms": synonyms,
                "source_chunk": source_chunk,
            }],
            collection_name,
        )
        return result.get(name)

    def _refine_entity(
        self,
        entity_name: str,
        entity_type: str,
        collection_name: str,
        source_chunk: int,
    ) -> Dict[str, Any]:
        """Refine a single entity using Qdrant.

        Args:
            entity_name: Name of the entity to refine
            entity_type: Type of the entity
            collection_name: Name of the Qdrant collection
            source_chunk: Source chunk ID

        Returns:
            Dictionary with refined entity information
        """
        # Query Qdrant for similar entities
        matches = self._query_qdrant(collection_name, entity_name, limit=5)

        # Determine canonical with LLM
        canonical_info = self._determine_canonical_with_llm(
            entity_name,
            matches,
            entity_type,
        )

        # Upsert canonical entity if it's new
        upserted_id = None
        if canonical_info["is_new"]:
            upserted_id = self._upsert_entity(
                collection_name=collection_name,
                name=canonical_info["canonical"],
                entity_type=entity_type,
                is_canonical=True,
                synonyms=canonical_info["synonyms"],
                source_chunk=source_chunk,
            )

        return {
            "original": entity_name,
            "canonical": canonical_info["canonical"],
            "canonical_id": upserted_id or canonical_info.get("canonical_id"),
            "synonyms": canonical_info["synonyms"],
            "is_new": canonical_info["is_new"],
        }

    def refine_triples(
        self,
        triples_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Refine all triples using Qdrant for entity resolution with batch processing.

        Args:
            triples_data: Dictionary containing triples data

        Returns:
            Refined triples data with canonical entities
        """
        # Ensure collections exist
        self._ensure_collections_exist()

        # Collect all unique entities
        entities_to_process = {
            "entity_registry": set(),  # For subjects and objects
            "predicate_registry": set(),  # For predicates
            "ontology_registry": set(),  # For entity types
        }

        for chunk_data in triples_data.get("chunks", []):
            for triple in chunk_data.get("triples", []):
                # Add subject entity
                entities_to_process["entity_registry"].add((
                    triple["subject"]["name"],
                    triple["subject"]["type"],
                ))

                # Add predicate
                entities_to_process["predicate_registry"].add((
                    triple["predicate"],
                    "predicate",
                ))

                # Add object entity
                entities_to_process["entity_registry"].add((
                    triple["object"]["name"],
                    triple["object"]["type"],
                ))

                # Add object type (ontology)
                entities_to_process["ontology_registry"].add((
                    triple["object"]["type"],
                    "ontology",
                ))

        # Process each collection in batch
        entity_cache = {}

        for collection_name, entities in entities_to_process.items():
            if not entities:
                continue

            print(f"Processing {len(entities)} entities in {collection_name}...")

            # Convert to list and get embeddings in batch
            entity_list = [{"name": name, "type": entity_type} for name, entity_type in entities]
            entity_names = [entity["name"] for entity in entity_list]

            # Get embeddings in batch
            print(f"  Getting embeddings for {len(entity_names)} entities...")
            embeddings = self._get_embeddings_batch(entity_names)

            # Process entities using similarity-based grouping
            print(f"  Grouping entities by similarity...")
            processed_canonicals = set()  # Track which canonicals we've already processed

            for i, (entity, embedding) in enumerate(zip(entity_list, embeddings)):
                # Skip if we've already processed this entity as part of a group
                if entity["name"] in processed_canonicals:
                    continue

                # Query Qdrant for similar entities (will find previously processed ones)
                matches = self._query_qdrant_with_embedding(
                    collection_name,
                    embedding,
                    limit=5,
                )

                # Find similar entities in current batch using cosine similarity
                similar_entities = []
                for j, (other_entity, other_embedding) in enumerate(zip(entity_list, embeddings)):
                    if i == j:
                        continue
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(embedding, other_embedding)
                    if similarity > 0.75:  # Similarity threshold
                        similar_entities.append({
                            "id": None,  # No ID yet, not in Qdrant
                            "score": similarity,
                            "payload": {
                                "name": other_entity["name"],
                                "type": other_entity["type"],
                            },
                        })

                # Combine Qdrant matches with similar entities from current batch
                all_matches = matches + similar_entities

                # Determine canonical with LLM (only if we have matches)
                if all_matches:
                    canonical_info = self._determine_canonical_with_llm(
                        entity["name"],
                        all_matches,
                        entity["type"],
                    )
                else:
                    # No matches, this entity is its own canonical
                    canonical_info = {
                        "canonical": entity["name"],
                        "canonical_id": None,
                        "synonyms": [],
                        "is_new": True,
                    }

                # Cache the result for the main entity
                entity_cache[(collection_name, entity["name"])] = {
                    "original": entity["name"],
                    "canonical": canonical_info["canonical"],
                    "canonical_id": canonical_info.get("canonical_id"),
                    "synonyms": canonical_info["synonyms"],
                    "is_new": canonical_info["is_new"],
                    "type": entity["type"],
                }

                # Mark this entity as processed
                processed_canonicals.add(entity["name"])

                # Cache results for similar entities that map to the same canonical
                for similar_entity in similar_entities:
                    similar_name = similar_entity["payload"]["name"]
                    if similar_name not in processed_canonicals:
                        entity_cache[(collection_name, similar_name)] = {
                            "original": similar_name,
                            "canonical": canonical_info["canonical"],
                            "canonical_id": None,  # Will be set when canonical is upserted
                            "synonyms": canonical_info["synonyms"],
                            "is_new": False,  # Not new, maps to existing canonical
                            "type": similar_entity["payload"]["type"],
                        }
                        processed_canonicals.add(similar_name)

                # If this is a new canonical entity, upsert it immediately
                if canonical_info["is_new"]:
                    upserted_id = self._batch_upsert_entities(
                        [{
                            "name": canonical_info["canonical"],
                            "type": entity["type"],
                            "is_canonical": True,
                            "synonyms": canonical_info["synonyms"],
                            "source_chunk": 1,
                        }],
                        collection_name,
                    )
                    # Update the cache with the upserted ID for all entities that map to this canonical
                    canonical_id = upserted_id.get(canonical_info["canonical"]) if upserted_id else None
                    if canonical_id:
                        for (cache_key, cached_entity) in entity_cache.items():
                            if (cache_key[0] == collection_name and
                                cached_entity["canonical"] == canonical_info["canonical"]):
                                cached_entity["canonical_id"] = canonical_id

        # Now refine all triples using the cached results
        refined_chunks = []

        for chunk_data in triples_data.get("chunks", []):
            chunk_id = chunk_data["chunk_id"]
            refined_triples = []

            for triple in chunk_data.get("triples", []):
                # Get refined subject
                subject_key = ("entity_registry", triple["subject"]["name"])
                refined_subject = entity_cache.get(subject_key, {
                    "original": triple["subject"]["name"],
                    "canonical": triple["subject"]["name"],
                    "canonical_id": None,
                    "synonyms": [],
                    "is_new": True,
                    "type": triple["subject"]["type"],
                })

                # Get refined predicate
                predicate_key = ("predicate_registry", triple["predicate"])
                refined_predicate = entity_cache.get(predicate_key, {
                    "original": triple["predicate"],
                    "canonical": triple["predicate"],
                    "canonical_id": None,
                    "synonyms": [],
                    "is_new": True,
                    "type": "predicate",
                })

                # Get refined object
                object_key = ("entity_registry", triple["object"]["name"])
                refined_object = entity_cache.get(object_key, {
                    "original": triple["object"]["name"],
                    "canonical": triple["object"]["name"],
                    "canonical_id": None,
                    "synonyms": [],
                    "is_new": True,
                    "type": triple["object"]["type"],
                })

                # Get refined object type
                object_type_key = ("ontology_registry", triple["object"]["type"])
                refined_object_type = entity_cache.get(object_type_key, {
                    "original": triple["object"]["type"],
                    "canonical": triple["object"]["type"],
                    "canonical_id": None,
                    "synonyms": [],
                    "is_new": True,
                    "type": "ontology",
                })

                # Create refined triple
                refined_triple = {
                    "subject": {
                        "name": refined_subject["canonical"],
                        "type": refined_subject["canonical_id"] or triple["subject"]["type"],
                        "original_name": triple["subject"]["name"],
                    },
                    "predicate": refined_predicate["canonical"],
                    "original_predicate": triple["predicate"],
                    "object": {
                        "name": refined_object["canonical"],
                        "type": refined_object_type["canonical"],
                        "original_name": triple["object"]["name"],
                        "original_type": triple["object"]["type"],
                    },
                    "properties": triple["properties"],
                    "chunk_id": chunk_id,
                    "refinement": {
                        "subject": refined_subject,
                        "predicate": refined_predicate,
                        "object": refined_object,
                        "object_type": refined_object_type,
                    },
                }

                refined_triples.append(refined_triple)

            refined_chunks.append({
                "chunk_id": chunk_id,
                "triples": refined_triples,
            })

        # Create refined output
        refined_data = {
            "source_file": triples_data.get("source_file"),
            "total_chunks": len(refined_chunks),
            "total_triples": sum(len(c["triples"]) for c in refined_chunks),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "refinement_applied": True,
            "chunks": refined_chunks,
        }

        return refined_data

    def save_refined_triples(
        self,
        refined_data: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Save refined triples to a JSON file.

        Args:
            refined_data: Refined triples data
            output_path: Path to save the refined triples

        Returns:
            Path to saved file
        """
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(refined_data, f, indent=2, ensure_ascii=False)

        return str(output_file)


def refine_triples_from_file(
    input_path: str,
    output_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> str:
    """Refine triples from a JSON file using Qdrant.

    Args:
        input_path: Path to input triples JSON file
        output_path: Path to save refined triples (default: input_path with _refined suffix)
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        llm_provider: LLM provider for canonical comparison
        llm_model: Model for LLM analysis

    Returns:
        Path to saved refined triples file
    """
    from pathlib import Path

    # Load input triples
    with open(input_path, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    # Create refiner
    refiner = TripleRefiner(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Refine triples
    refined_data = refiner.refine_triples(triples_data)

    # Generate output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_refined{input_file.suffix}"

    # Save refined triples
    result_path = refiner.save_refined_triples(refined_data, output_path)

    return result_path
