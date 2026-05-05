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
        similarity_threshold: float = 0.85,
    ):
        """Initialize the triple refiner.

        Args:
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            llm_provider: LLM provider for canonical comparison
            llm_model: Model for LLM analysis
            registry_info_dir: Directory containing registry specification files
            similarity_threshold: Cosine similarity threshold for entity matching (0.0-1.0)
        """
        import os
        from pathlib import Path

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold

        # Set registry info directory
        if registry_info_dir is None:
            # Default to registry_info in the project root
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            self.registry_info_dir = project_root / "registry_info"

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
        collections = ["entity_registry", "predicate_registry", "label_registry"]

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
            "label_registry": "Ontology registry for entity types",
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

            # Ensure payload index exists for is_canonical (required for filtered search)
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="is_canonical",
                    field_schema=qdrant_models.PayloadSchemaType.BOOL,
                )
                print(f"  Created is_canonical index on {collection_name}")
            except Exception:
                pass  # Index already exists

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

    def _query_qdrant_canonical(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        """Query Qdrant for canonical entities above the similarity threshold.

        Args:
            collection_name: Name of the collection to query
            query_vector: Pre-computed embedding vector
            limit: Maximum number of results to return

        Returns:
            List of matching canonical points with payloads
        """
        try:
            from qdrant_client.http import models as qdrant_models

            vector_name = self._get_vector_name(collection_name)

            search_results = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
                score_threshold=self.similarity_threshold,
                query_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="is_canonical",
                            match=qdrant_models.MatchValue(value=True),
                        )
                    ]
                ),
            )

            matches = []
            for result in search_results.points:
                matches.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })

            return matches

        except Exception as e:
            print(f"Warning: Failed to query Qdrant for canonicals: {e}")
            return []

    def _compare_canonical_with_llm(
        self,
        new_entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        entity_type: str,
    ) -> Dict[str, Any]:
        """Use LLM to decide whether the new entity should replace the existing canonical.

        Args:
            new_entity: New entity dict (name, name_en, attributes, attributes_en, label)
            existing_canonical: Qdrant match dict (id, score, payload)
            entity_type: Type/label of entity

        Returns:
            {"are_same_entity": bool, "new_is_canonical": bool, "reasoning": str}
        """
        prompt = self._create_canonical_comparison_prompt(
            new_entity,
            existing_canonical,
            entity_type,
        )
        response = self._get_llm_response(prompt)
        return self._parse_canonical_winner(response)

    def _compare_predicate_with_llm(
        self,
        new_predicate: str,
        new_predicate_en: Optional[str],
        existing_canonical: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to decide whether two predicates are semantically equivalent.

        Predicates need a different strategy from entities: "increase", "grow",
        and "improve" are all semantically equivalent as relationship types even
        though the words are different.  When they are equivalent the more
        standard/generic English form should become (or stay) the canonical.

        Args:
            new_predicate: Incoming predicate name (may be non-English)
            new_predicate_en: English translation of the incoming predicate
            existing_canonical: Qdrant match dict (id, score, payload)

        Returns:
            {"are_equivalent": bool, "new_is_canonical": bool, "canonical_form": str, "reasoning": str}
        """
        old_payload = existing_canonical.get("payload", {})
        existing_name = old_payload.get("name_en") or old_payload.get("name", "")
        incoming_name = new_predicate_en or new_predicate

        prompt = f"""You are an expert in knowledge graph schema design and relationship normalisation.

Two predicates (relationship types) have high vector similarity.  Your task:

**Step 1 – Semantic equivalence check:** Decide whether these two predicates express the
*same relationship type* in a knowledge graph.  Synonyms and near-synonyms count as
equivalent (e.g. "increase" / "grow" / "rise" / "improve" are all equivalent; but
"increase" and "decrease" are NOT equivalent even though they are related).

**Step 2 – Canonical form selection (only when equivalent):** Choose the most standard,
concise, and generic English verb form as the canonical predicate.  Prefer:
- Short, common English verbs over long phrases
- Active voice, base form (infinitive without "to")
- Domain-neutral wording when both options work

**Existing Canonical:**  {existing_name}
  (original: {old_payload.get("name", "")})
  ID: {existing_canonical.get("id")}
  Similarity score: {existing_canonical.get("score", 0):.3f}

**Incoming Predicate:**  {incoming_name}
  (original: {new_predicate})

**Output Format (JSON):**
```json
{{
  "are_equivalent": true_or_false,
  "new_is_canonical": true_or_false,
  "canonical_form": "chosen canonical predicate string",
  "reasoning": "brief explanation"
}}
```

Rules:
- If "are_equivalent" is false → set "new_is_canonical" to true so the incoming predicate
  gets its own canonical entry; "canonical_form" should be {incoming_name!r}.
- If "are_equivalent" is true → set "new_is_canonical" based on which form is superior;
  "canonical_form" must be the chosen canonical string.

Return only valid JSON:"""

        response = self._get_llm_response(prompt)
        return self._parse_predicate_equivalence(response, incoming_name)

    def _parse_predicate_equivalence(
        self,
        response: str,
        fallback_name: str,
    ) -> Dict[str, Any]:
        """Parse LLM response for predicate equivalence decision.

        Args:
            response: LLM response text
            fallback_name: Name to use if parsing fails

        Returns:
            {"are_equivalent": bool, "new_is_canonical": bool, "canonical_form": str, "reasoning": str}
        """
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)
            are_equiv = bool(data.get("are_equivalent", True))
            new_is_canonical = bool(data.get("new_is_canonical", False)) if are_equiv else True
            return {
                "are_equivalent": are_equiv,
                "new_is_canonical": new_is_canonical,
                "canonical_form": data.get("canonical_form", fallback_name),
                "reasoning": data.get("reasoning", ""),
            }
        except Exception as e:
            print(f"Warning: Failed to parse predicate equivalence response: {e}")
            return {
                "are_equivalent": True,
                "new_is_canonical": False,
                "canonical_form": fallback_name,
                "reasoning": "parse error",
            }

    def _create_canonical_comparison_prompt(
        self,
        new_entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        entity_type: str,
    ) -> str:
        """Create prompt to decide whether two entities are the same and, if so, which is canonical.

        Args:
            new_entity: New entity dict
            existing_canonical: Existing canonical Qdrant match
            entity_type: Type/label of entity

        Returns:
            Prompt string
        """
        def _format_entity(name: str, name_en: Optional[str], attributes: Dict) -> str:
            lines = [f"  Name: {name}"]
            if name_en:
                lines.append(f"  Name (EN): {name_en}")
            if attributes:
                lines.append(f"  Evidence/Attributes: {json.dumps(attributes, ensure_ascii=False)}")
            return "\n".join(lines)

        old_payload = existing_canonical.get("payload", {})
        old_text = _format_entity(
            old_payload.get("name", ""),
            old_payload.get("name_en"),
            old_payload.get("attributes") or {},
        )

        new_text = _format_entity(
            new_entity.get("name", ""),
            new_entity.get("name_en"),
            new_entity.get("attributes") or {},
        )

        prompt = f"""You are an expert in entity resolution and knowledge graph curation.

Two entities with high vector similarity have been found.  Your task has two steps:

**Step 1 – Identity check:** Decide whether these two entities refer to the *same real-world
concept* (i.e. they are aliases, abbreviations, or surface variants of the same thing).
Set "are_same_entity": true ONLY when both names clearly denote the identical concept.
If one is a specific attribute, metric, sub-component, or related-but-distinct concept of the
other (e.g. "Thai E-Commerce Market" vs "Thai E-Commerce Market Annual Value"), they are
DIFFERENT entities — set "are_same_entity": false.

**Step 2 – Canonical selection (only when are_same_entity is true):** Decide which form should
be the canonical entry.  The existing canonical keeps its status UNLESS the new entity is clearly
superior (more specific name, better evidence, more complete information).

**Entity Type:** {entity_type}

**Existing Canonical (currently registered):**
{old_text}
  ID: {existing_canonical.get("id")}
  Similarity score: {existing_canonical.get("score", 0):.3f}

**New Entity (incoming):**
{new_text}

**Output Format (JSON):**
```json
{{
  "are_same_entity": true_or_false,
  "new_is_canonical": true_or_false,
  "reasoning": "brief explanation"
}}
```

Rules:
- If "are_same_entity" is false, set "new_is_canonical" to true (the new entity needs its own canonical entry).
- If "are_same_entity" is true, set "new_is_canonical" based on which form is superior.

Return only valid JSON:"""

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

    def _parse_canonical_winner(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """Parse LLM response for canonical winner decision.

        Args:
            response: LLM response text

        Returns:
            {"are_same_entity": bool, "new_is_canonical": bool, "reasoning": str}
        """
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            data = json.loads(response)
            are_same = bool(data.get("are_same_entity", True))
            # When entities are distinct, the new entity must become its own canonical
            new_is_canonical = bool(data.get("new_is_canonical", False)) if are_same else True
            return {
                "are_same_entity": are_same,
                "new_is_canonical": new_is_canonical,
                "reasoning": data.get("reasoning", ""),
            }

        except Exception as e:
            print(f"Warning: Failed to parse canonical winner response: {e}")
            # Fallback: keep existing canonical, assume same entity
            return {"are_same_entity": True, "new_is_canonical": False, "reasoning": "parse error"}

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
            entities: List of entity dictionaries with name, name_en, label, attributes, etc.
                      When name_en is present, it is used for both the embedding and the point
                      UUID so that queries in either language resolve to the same point.
            collection_name: Name of the collection

        Returns:
            Dictionary mapping entity names (original) to point IDs
        """
        if not entities:
            return {}

        try:
            from qdrant_client.http import models as qdrant_models

            vector_config = self._get_vector_config(collection_name)
            vector_name = self._get_vector_name(collection_name)

            # Embed using the English name when available so that both a Thai query
            # ("ทุนทางสังคม") and its English translation ("Social Capital") land on
            # the same point via multilingual similarity.
            embed_texts = [entity.get("name_en") or entity["name"] for entity in entities]
            embeddings = self._get_embeddings_batch(embed_texts)

            points = []
            for entity, embedding in zip(entities, embeddings):
                # Derive the canonical key used for UUID from the English name when
                # available, so English queries return the exact same point ID.
                canonical_key = entity.get("name_en") or entity["name"]
                point_id = self._generate_uuid(canonical_key)

                payload: Dict[str, Any] = {
                    "name": entity["name"],
                    "is_canonical": entity.get("is_canonical", True),
                }

                # Link to canonical when this entity is not canonical itself
                if entity.get("canonical_id") is not None:
                    payload["canonical_id"] = entity["canonical_id"]

                # English translation — enables bilingual search
                if entity.get("name_en"):
                    payload["name_en"] = entity["name_en"]

                # Semantic label (replaces the old "type" field)
                label = entity.get("label") or entity.get("type", "")
                if label:
                    payload["label"] = label
                # Keep "type" as an alias for backward compatibility
                payload["type"] = label

                # Unbounded attribute dictionaries
                if entity.get("attributes"):
                    payload["attributes"] = entity["attributes"]
                if entity.get("attributes_en"):
                    payload["attributes_en"] = entity["attributes_en"]

                point = qdrant_models.PointStruct(
                    id=point_id,
                    vector={vector_name: embedding},
                    payload=payload,
                )
                points.append(point)

            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )

            # Return mapping: original name → point ID
            result = {}
            for entity in entities:
                canonical_key = entity.get("name_en") or entity["name"]
                result[entity["name"]] = self._generate_uuid(canonical_key)
            return result

        except Exception as e:
            print(f"Error: Failed to batch upsert entities to {collection_name}: {e}")
            print(f"  Entities being upserted: {[e['name'] for e in entities]}")
            raise

    def _upsert_entity(
        self,
        collection_name: str,
        name: str,
        entity_type: str,
        is_canonical: bool,
        canonical_id: Optional[str] = None,
        name_en: Optional[str] = None,
        label: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        attributes_en: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upsert a single entity to Qdrant.

        Args:
            collection_name: Name of the collection
            name: Entity name (original language)
            entity_type: Type/label of entity
            is_canonical: Whether this is the canonical form
            canonical_id: ID of the canonical entity (when is_canonical is False)
            name_en: English translation of the name (enables bilingual search)
            label: Semantic PascalCase label (replaces entity_type when provided)
            attributes: Unbounded attribute dict (original language)
            attributes_en: Unbounded attribute dict (English)
        Returns:
            ID of the upserted point
        """
        entity: Dict[str, Any] = {
            "name": name,
            "type": entity_type,
            "label": label or entity_type,
            "is_canonical": is_canonical,
        }
        if canonical_id is not None:
            entity["canonical_id"] = canonical_id
        if name_en:
            entity["name_en"] = name_en
        if attributes:
            entity["attributes"] = attributes
        if attributes_en:
            entity["attributes_en"] = attributes_en

        result = self._batch_upsert_entities([entity], collection_name)
        return result.get(name)

    def _refine_entity(
        self,
        entity_name: str,
        entity_type: str,
        collection_name: str,
    ) -> Dict[str, Any]:
        """Refine a single entity using Qdrant.

        Args:
            entity_name: Name of the entity to refine
            entity_type: Type of the entity
            collection_name: Name of the Qdrant collection

        Returns:
            Dictionary with refined entity information
        """
        embedding = self._get_embedding(entity_name)
        canonical_matches = self._query_qdrant_canonical(collection_name, embedding, limit=1)

        if not canonical_matches:
            upserted_id = self._upsert_entity(
                collection_name=collection_name,
                name=entity_name,
                entity_type=entity_type,
                is_canonical=True,
            )
            return {
                "original": entity_name,
                "canonical": entity_name,
                "canonical_id": upserted_id,
                "is_new": True,
                "is_canonical": True,
            }

        existing_canonical = canonical_matches[0]
        winner = self._compare_canonical_with_llm(
            new_entity={"name": entity_name, "label": entity_type},
            existing_canonical=existing_canonical,
            entity_type=entity_type,
        )

        if not winner.get("are_same_entity", True):
            # Distinct entities — create a new canonical
            upserted_id = self._upsert_entity(
                collection_name=collection_name,
                name=entity_name,
                entity_type=entity_type,
                is_canonical=True,
            )
            return {
                "original": entity_name,
                "canonical": entity_name,
                "canonical_id": upserted_id,
                "is_new": True,
                "is_canonical": True,
            }
        elif not winner["new_is_canonical"]:
            upserted_id = self._upsert_entity(
                collection_name=collection_name,
                name=entity_name,
                entity_type=entity_type,
                is_canonical=False,
                canonical_id=existing_canonical["id"],
            )
            return {
                "original": entity_name,
                "canonical": existing_canonical["payload"].get("name", entity_name),
                "canonical_id": existing_canonical["id"],
                "is_new": False,
                "is_canonical": False,
            }
        else:
            upserted_id = self._upsert_entity(
                collection_name=collection_name,
                name=entity_name,
                entity_type=entity_type,
                is_canonical=True,
            )
            old_id = existing_canonical["id"]
            try:
                self.qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={"is_canonical": False, "canonical_id": upserted_id},
                    points=[old_id],
                )
            except Exception as e:
                print(f"  Warning: Failed to demote old canonical: {e}")
            return {
                "original": entity_name,
                "canonical": entity_name,
                "canonical_id": upserted_id,
                "is_new": True,
                "is_canonical": True,
            }

    def refine_triples(
        self,
        triples_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Refine all triples using Qdrant for entity resolution with batch processing.

        Supports the updated triple format which uses:
        - subject/object ``label`` (semantic PascalCase) instead of ``type``
        - ``name_en`` / ``predicate_en`` English translations on non-English sources
        - Unbounded ``attributes`` / ``attributes_en`` dicts on subjects and objects
        - ``relationship_attributes`` / ``relationship_attributes_en`` on the edge

        Bilingual search strategy: when a ``name_en`` translation is present the
        English text is used for both the embedding and the point UUID.  This means
        a Qdrant query using the Thai original *or* its English translation will
        resolve to the same point because (a) the stored vector is the English
        embedding and (b) OpenAI's ``text-embedding-3-small`` is cross-lingual, so
        semantically equivalent Thai text maps to a very similar vector.

        Args:
            triples_data: Dictionary containing triples data

        Returns:
            Refined triples data with canonical entities
        """
        # Ensure collections exist
        self._ensure_collections_exist()

        # ── Collect unique entities, keeping the full new-format metadata ─────
        # Use dicts keyed by original name so we preserve name_en / label /
        # attributes without losing information when the same name appears in
        # multiple triples.
        entity_registry_items: Dict[str, Dict[str, Any]] = {}
        predicate_registry_items: Dict[str, Dict[str, Any]] = {}
        label_registry_items: Dict[str, Dict[str, Any]] = {}

        for chunk_data in triples_data.get("chunks", []):
            for triple in chunk_data.get("triples", []):
                subj = triple["subject"]
                obj = triple["object"]

                # Subject
                subj_name = subj["name"]
                if subj_name not in entity_registry_items:
                    entity_registry_items[subj_name] = {
                        "name": subj_name,
                        "name_en": subj.get("name_en"),
                        "label": subj.get("label", ""),
                        "attributes": subj.get("attributes", {}),
                        "attributes_en": subj.get("attributes_en", {}),
                    }

                # Predicate
                pred = triple["predicate"]
                pred_en = triple.get("predicate_en")
                if pred not in predicate_registry_items:
                    predicate_registry_items[pred] = {
                        "name": pred,
                        # predicate_en is already ALL_CAPS_SNAKE_CASE English
                        "name_en": pred_en,
                        "label": "predicate",
                    }

                # Object
                obj_name = obj["name"]
                if obj_name not in entity_registry_items:
                    entity_registry_items[obj_name] = {
                        "name": obj_name,
                        "name_en": obj.get("name_en"),
                        "label": obj.get("label", ""),
                        "attributes": obj.get("attributes", {}),
                        "attributes_en": obj.get("attributes_en", {}),
                    }

                # Labels → label_registry (labels are already English PascalCase)
                for lbl in (subj.get("label", ""), obj.get("label", "")):
                    if lbl and lbl not in label_registry_items:
                        label_registry_items[lbl] = {
                            "name": lbl,
                            "name_en": lbl,  # already English
                            "label": "label",
                        }

        # ── Process each registry collection ──────────────────────────────────
        entity_cache: Dict[tuple, Dict[str, Any]] = {}
        # Log of canonical changes: old_canonical_id → new_canonical_id
        canonical_change_log: List[Dict[str, Any]] = []

        collections = {
            "entity_registry": list(entity_registry_items.values()),
            "predicate_registry": list(predicate_registry_items.values()),
            "label_registry": list(label_registry_items.values()),
        }

        for collection_name, entity_list in collections.items():
            if not entity_list:
                continue

            print(f"Processing {len(entity_list)} entities in {collection_name}...")

            embed_texts = [e.get("name_en") or e["name"] for e in entity_list]
            print(f"  Getting embeddings for {len(embed_texts)} entities...")
            embeddings = self._get_embeddings_batch(embed_texts)

            for entity, embedding in zip(entity_list, embeddings):
                # Query Qdrant for existing canonicals at >= similarity_threshold
                canonical_matches = self._query_qdrant_canonical(
                    collection_name,
                    embedding,
                    limit=1,
                )

                if not canonical_matches:
                    # No canonical exists → this entity becomes canonical
                    upserted_ids = self._batch_upsert_entities(
                        [{
                            "name": entity["name"],
                            "name_en": entity.get("name_en"),
                            "label": entity.get("label", ""),
                            "attributes": entity.get("attributes", {}),
                            "attributes_en": entity.get("attributes_en", {}),
                            "is_canonical": True,
                        }],
                        collection_name,
                    )
                    canonical_id = upserted_ids.get(entity["name"])
                    print(f"  New canonical: '{entity['name']}' ({canonical_id})")
                    entity_cache[(collection_name, entity["name"])] = {
                        "original": entity["name"],
                        "canonical": entity["name"],
                        "canonical_id": canonical_id,
                        "is_new": True,
                        "is_canonical": True,
                        "label": entity.get("label", ""),
                        "name_en": entity.get("name_en"),
                        "attributes": entity.get("attributes", {}),
                        "attributes_en": entity.get("attributes_en", {}),
                    }

                else:
                    existing_canonical = canonical_matches[0]

                    if collection_name == "predicate_registry":
                        # ── Predicate-specific semantic equivalence check ──────
                        winner = self._compare_predicate_with_llm(
                            new_predicate=entity["name"],
                            new_predicate_en=entity.get("name_en"),
                            existing_canonical=existing_canonical,
                        )

                        if not winner["are_equivalent"]:
                            # Distinct predicate — create its own canonical entry
                            upserted_ids = self._batch_upsert_entities(
                                [{
                                    "name": entity["name"],
                                    "name_en": entity.get("name_en"),
                                    "label": entity.get("label", "predicate"),
                                    "is_canonical": True,
                                }],
                                collection_name,
                            )
                            canonical_id = upserted_ids.get(entity["name"])
                            print(
                                f"  New predicate canonical (distinct from "
                                f"'{existing_canonical['payload'].get('name')}'): "
                                f"'{entity['name']}' ({canonical_id})"
                            )
                            entity_cache[(collection_name, entity["name"])] = {
                                "original": entity["name"],
                                "canonical": entity["name"],
                                "canonical_id": canonical_id,
                                "is_new": True,
                                "is_canonical": True,
                                "label": "predicate",
                                "name_en": entity.get("name_en"),
                                "attributes": {},
                                "attributes_en": {},
                            }

                        elif not winner["new_is_canonical"]:
                            # Equivalent — existing stays canonical; map incoming to it
                            upserted_ids = self._batch_upsert_entities(
                                [{
                                    "name": entity["name"],
                                    "name_en": entity.get("name_en"),
                                    "label": "predicate",
                                    "is_canonical": False,
                                    "canonical_id": existing_canonical["id"],
                                }],
                                collection_name,
                            )
                            canonical_name = existing_canonical["payload"].get("name_en") \
                                or existing_canonical["payload"].get("name", entity["name"])
                            print(
                                f"  Predicate '{entity['name']}' merged into canonical "
                                f"'{canonical_name}'"
                            )
                            entity_cache[(collection_name, entity["name"])] = {
                                "original": entity["name"],
                                "canonical": canonical_name,
                                "canonical_id": existing_canonical["id"],
                                "is_new": False,
                                "is_canonical": False,
                                "label": "predicate",
                                "name_en": entity.get("name_en"),
                                "attributes": {},
                                "attributes_en": {},
                            }

                        else:
                            # Equivalent — new predicate is the better canonical form
                            canonical_form = winner.get("canonical_form") or entity.get("name_en") or entity["name"]
                            upserted_ids = self._batch_upsert_entities(
                                [{
                                    "name": entity["name"],
                                    "name_en": entity.get("name_en"),
                                    "label": "predicate",
                                    "is_canonical": True,
                                }],
                                collection_name,
                            )
                            new_id = upserted_ids.get(entity["name"])
                            old_id = existing_canonical["id"]

                            # Demote old canonical
                            try:
                                self.qdrant_client.set_payload(
                                    collection_name=collection_name,
                                    payload={"is_canonical": False, "canonical_id": new_id},
                                    points=[old_id],
                                )
                            except Exception as e:
                                print(f"  Warning: Failed to demote old predicate canonical: {e}")

                            canonical_change_log.append({
                                "collection": collection_name,
                                "old_canonical_id": old_id,
                                "old_canonical_name": existing_canonical["payload"].get("name"),
                                "new_canonical_id": new_id,
                                "new_canonical_name": entity["name"],
                                "reasoning": winner.get("reasoning", ""),
                            })
                            print(
                                f"  Predicate canonical replaced: "
                                f"'{existing_canonical['payload'].get('name')}' "
                                f"({old_id}) → '{entity['name']}' ({new_id})"
                            )
                            entity_cache[(collection_name, entity["name"])] = {
                                "original": entity["name"],
                                "canonical": canonical_form,
                                "canonical_id": new_id,
                                "is_new": True,
                                "is_canonical": True,
                                "label": "predicate",
                                "name_en": entity.get("name_en"),
                                "attributes": {},
                                "attributes_en": {},
                            }

                    else:
                        # ── Entity / label registry: identity-first check ──────
                        winner = self._compare_canonical_with_llm(
                            new_entity=entity,
                            existing_canonical=existing_canonical,
                            entity_type=entity.get("label", collection_name),
                        )

                        if not winner.get("are_same_entity", True):
                            # Distinct entities — create a new canonical for the incoming entity
                            upserted_ids = self._batch_upsert_entities(
                                [{
                                    "name": entity["name"],
                                    "name_en": entity.get("name_en"),
                                    "label": entity.get("label", ""),
                                    "attributes": entity.get("attributes", {}),
                                    "attributes_en": entity.get("attributes_en", {}),
                                    "is_canonical": True,
                                }],
                                collection_name,
                            )
                            canonical_id = upserted_ids.get(entity["name"])
                            print(
                                f"  New canonical (distinct from '{existing_canonical['payload'].get('name')}'): "
                                f"'{entity['name']}' ({canonical_id})"
                            )
                            entity_cache[(collection_name, entity["name"])] = {
                                "original": entity["name"],
                                "canonical": entity["name"],
                                "canonical_id": canonical_id,
                                "is_new": True,
                                "is_canonical": True,
                                "label": entity.get("label", ""),
                                "name_en": entity.get("name_en"),
                                "attributes": entity.get("attributes", {}),
                                "attributes_en": entity.get("attributes_en", {}),
                            }

                        elif not winner["new_is_canonical"]:
                            # Same entity — old stays canonical → register new as non-canonical
                            upserted_ids = self._batch_upsert_entities(
                                [{
                                    "name": entity["name"],
                                    "name_en": entity.get("name_en"),
                                    "label": entity.get("label", ""),
                                    "attributes": entity.get("attributes", {}),
                                    "attributes_en": entity.get("attributes_en", {}),
                                    "is_canonical": False,
                                    "canonical_id": existing_canonical["id"],
                                }],
                                collection_name,
                            )
                            print(
                                f"  '{entity['name']}' registered under existing canonical "
                                f"'{existing_canonical['payload'].get('name')}'"
                            )
                            entity_cache[(collection_name, entity["name"])] = {
                                "original": entity["name"],
                                "canonical": existing_canonical["payload"].get("name", entity["name"]),
                                "canonical_id": existing_canonical["id"],
                                "is_new": False,
                                "is_canonical": False,
                                "label": entity.get("label", ""),
                                "name_en": entity.get("name_en"),
                                "attributes": entity.get("attributes", {}),
                                "attributes_en": entity.get("attributes_en", {}),
                            }

                        else:
                            # Same entity — new entity becomes canonical
                            upserted_ids = self._batch_upsert_entities(
                                [{
                                    "name": entity["name"],
                                    "name_en": entity.get("name_en"),
                                    "label": entity.get("label", ""),
                                    "attributes": entity.get("attributes", {}),
                                    "attributes_en": entity.get("attributes_en", {}),
                                    "is_canonical": True,
                                }],
                                collection_name,
                            )
                            new_id = upserted_ids.get(entity["name"])
                            old_id = existing_canonical["id"]

                            # Demote old canonical
                            try:
                                self.qdrant_client.set_payload(
                                    collection_name=collection_name,
                                    payload={"is_canonical": False, "canonical_id": new_id},
                                    points=[old_id],
                                )
                            except Exception as e:
                                print(f"  Warning: Failed to demote old canonical: {e}")

                            # Log the change for later refinement
                            canonical_change_log.append({
                                "collection": collection_name,
                                "old_canonical_id": old_id,
                                "old_canonical_name": existing_canonical["payload"].get("name"),
                                "new_canonical_id": new_id,
                                "new_canonical_name": entity["name"],
                                "reasoning": winner.get("reasoning", ""),
                            })
                            print(
                                f"  Canonical replaced: '{existing_canonical['payload'].get('name')}' "
                                f"({old_id}) → '{entity['name']}' ({new_id})"
                            )

                            entity_cache[(collection_name, entity["name"])] = {
                                "original": entity["name"],
                                "canonical": entity["name"],
                                "canonical_id": new_id,
                                "is_new": True,
                                "is_canonical": True,
                                "label": entity.get("label", ""),
                                "name_en": entity.get("name_en"),
                                "attributes": entity.get("attributes", {}),
                                "attributes_en": entity.get("attributes_en", {}),
                            }

        # ── Build refined triples ──────────────────────────────────────────────
        refined_chunks = []

        for chunk_data in triples_data.get("chunks", []):
            chunk_id = chunk_data["chunk_id"]
            refined_triples = []

            for triple in chunk_data.get("triples", []):
                subj_raw = triple["subject"]
                obj_raw = triple["object"]

                # Refined subject
                subject_key = ("entity_registry", subj_raw["name"])
                refined_subject = entity_cache.get(subject_key, {
                    "original": subj_raw["name"],
                    "canonical": subj_raw["name"],
                    "canonical_id": None,
                    "is_new": True,
                    "label": subj_raw.get("label", ""),
                    "name_en": subj_raw.get("name_en"),
                    "attributes": subj_raw.get("attributes", {}),
                    "attributes_en": subj_raw.get("attributes_en", {}),
                })

                # Refined predicate
                predicate_key = ("predicate_registry", triple["predicate"])
                refined_predicate = entity_cache.get(predicate_key, {
                    "original": triple["predicate"],
                    "canonical": triple["predicate"],
                    "canonical_id": None,
                    "is_new": True,
                    "label": "predicate",
                    "name_en": triple.get("predicate_en"),
                })

                # Refined object
                object_key = ("entity_registry", obj_raw["name"])
                refined_object = entity_cache.get(object_key, {
                    "original": obj_raw["name"],
                    "canonical": obj_raw["name"],
                    "canonical_id": None,
                    "is_new": True,
                    "label": obj_raw.get("label", ""),
                    "name_en": obj_raw.get("name_en"),
                    "attributes": obj_raw.get("attributes", {}),
                    "attributes_en": obj_raw.get("attributes_en", {}),
                })

                # Refined object label (ontology registry)
                obj_label_raw = obj_raw.get("label", "")
                object_label_key = ("label_registry", obj_label_raw)
                refined_object_label = entity_cache.get(object_label_key, {
                    "original": obj_label_raw,
                    "canonical": obj_label_raw,
                    "canonical_id": None,
                    "is_new": True,
                    "label": "label",
                })

                # Canonical label for the object (fall back through layers)
                canonical_obj_label = (
                    refined_object_label["canonical"]
                    or refined_object.get("label", "")
                    or obj_label_raw
                )

                refined_triple: Dict[str, Any] = {
                    "subject": {
                        "name": refined_subject["canonical"],
                        "original_name": subj_raw["name"],
                        "label": refined_subject.get("label", ""),
                        # type kept for backward compatibility with neo4j_graph_builder
                        "type": refined_subject.get("label", ""),
                    },
                    "predicate": refined_predicate["canonical"],
                    "original_predicate": triple["predicate"],
                    "object": {
                        "name": refined_object["canonical"],
                        "original_name": obj_raw["name"],
                        "label": canonical_obj_label,
                        # type kept for backward compatibility
                        "type": canonical_obj_label,
                        "original_label": obj_label_raw,
                    },
                    "properties": triple.get("properties", {}),
                    "chunk_id": chunk_id,
                    "refinement": {
                        "subject": refined_subject,
                        "predicate": refined_predicate,
                        "object": refined_object,
                        "object_label": refined_object_label,
                    },
                }

                # ── Optional English-translation fields (new format) ──────────
                if refined_subject.get("name_en"):
                    refined_triple["subject"]["name_en"] = refined_subject["name_en"]
                if subj_raw.get("attributes"):
                    refined_triple["subject"]["attributes"] = subj_raw["attributes"]
                if subj_raw.get("attributes_en"):
                    refined_triple["subject"]["attributes_en"] = subj_raw["attributes_en"]

                pred_en = triple.get("predicate_en") or refined_predicate.get("name_en")
                if pred_en:
                    refined_triple["predicate_en"] = pred_en

                if refined_object.get("name_en"):
                    refined_triple["object"]["name_en"] = refined_object["name_en"]
                if obj_raw.get("attributes"):
                    refined_triple["object"]["attributes"] = obj_raw["attributes"]
                if obj_raw.get("attributes_en"):
                    refined_triple["object"]["attributes_en"] = obj_raw["attributes_en"]

                if triple.get("relationship_attributes"):
                    refined_triple["relationship_attributes"] = triple["relationship_attributes"]
                if triple.get("relationship_attributes_en"):
                    refined_triple["relationship_attributes_en"] = triple["relationship_attributes_en"]

                refined_triples.append(refined_triple)

            refined_chunks.append({
                "chunk_id": chunk_id,
                "triples": refined_triples,
            })

        refined_data: Dict[str, Any] = {
            "source_file": triples_data.get("source_file"),
            "total_chunks": len(refined_chunks),
            "total_triples": sum(len(c["triples"]) for c in refined_chunks),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "refinement_applied": True,
            "canonical_change_log": canonical_change_log,
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
    similarity_threshold: float = 0.85,
) -> str:
    """Refine triples from a JSON file using Qdrant.

    Args:
        input_path: Path to input triples JSON file
        output_path: Path to save refined triples (default: input_path with _refined suffix)
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        llm_provider: LLM provider for canonical comparison
        llm_model: Model for LLM analysis
        similarity_threshold: Cosine similarity threshold for entity matching (0.0-1.0)

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
        similarity_threshold=similarity_threshold,
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
