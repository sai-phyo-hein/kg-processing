"""Triple refinement module for entity resolution using Qdrant.

Single-vector architecture
--------------------------
Every point in Qdrant carries **one named dense vector** whose name is
derived from the collection's ``vector_name`` field in its registry spec JSON:

  • ``{vector_name}``  – OpenAI embedding of the entity label (English).

Per-collection vector names (from registry spec JSON files):
  entity_registry    → entity    (sparse: entity_vector)
  predicate_registry → predicate (sparse: predicate_vector)
  label_registry     → label     (sparse: label_vector)

All labels, entity names, and predicates are in English.  Qdrant queries
use the single vector per collection for entity resolution.

Point UUID
----------
Derived deterministically from the entity name so that the same real-world
concept always maps to the same Qdrant point.

Payload schema
--------------
    name         – entity label / surface form (English)
    label        – semantic PascalCase ontology label
    type         – alias for label (backward-compat)
    is_canonical – bool; True  = this point IS the canonical entry
    canonical_id – UUID of the canonical point (when is_canonical=False)
    attributes   – free-form dict
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_SIZE = 1536


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
        """Initialise the triple refiner.

        Args:
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            llm_provider: LLM provider for canonical comparison
            llm_model: Model for LLM analysis
            registry_info_dir: Directory containing registry specification files
            similarity_threshold: Cosine similarity threshold for entity matching
        """
        import os
        from pathlib import Path

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.similarity_threshold = similarity_threshold

        if registry_info_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            self.registry_info_dir = project_root / "registry_info"
        else:
            self.registry_info_dir = Path(registry_info_dir)

        self.registry_specs = self._load_registry_specs()

        print(f"Registry info directory: {self.registry_info_dir}")
        print(f"Registry specs loaded: {list(self.registry_specs.keys())}")

        self._init_qdrant_client()

    # ------------------------------------------------------------------
    # Qdrant client
    # ------------------------------------------------------------------

    def _init_qdrant_client(self):
        """Initialise Qdrant client."""
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
            raise RuntimeError(f"Failed to initialise Qdrant client: {e}")

    # ------------------------------------------------------------------
    # Registry spec helpers
    # ------------------------------------------------------------------

    def _load_registry_specs(self) -> Dict[str, Dict[str, Any]]:
        """Load registry specifications by scanning the registry_info directory.

        Every ``*.json`` file in the directory defines one registry.  The file
        stem (e.g. ``entity_registry``) becomes the internal *spec key* used
        throughout the code; the ``collection_name`` field inside the file is
        the actual Qdrant collection name.

        Recognised spec fields (all defined in the registry JSON files):
          collection_name    – actual Qdrant collection name (defaults to file stem)
          triple_role        – which part of a triple this registry handles:
                               ``"entity"``, ``"predicate"``, ``"label"``, or ``"metadata"``
          vector_name        – base name; ``_en`` / ``_th`` suffixes are added
                               to form the two named dense vectors per collection
          sparse_vector_name – name of the sparse vector index (kept as-is)
          vector_config      – size, distance, hnsw_config, datatype, on_disk
        """
        specs: Dict[str, Dict[str, Any]] = {}
        self._role_to_spec_key: Dict[str, str] = {}

        spec_files = sorted(self.registry_info_dir.glob("*.json"))
        if not spec_files:
            print(f"Warning: No registry spec files found in {self.registry_info_dir}")

        for spec_file in spec_files:
            spec_key = spec_file.stem  # e.g. "entity_registry"
            with open(spec_file, "r") as f:
                spec = json.load(f)
            # Ensure collection_name falls back to the file stem
            spec.setdefault("collection_name", spec_key)
            specs[spec_key] = spec

            role = spec.get("triple_role")
            if role:
                self._role_to_spec_key[role] = spec_key
                print(f"Registry '{spec_key}' → role '{role}' → Qdrant '{spec['collection_name']}'")

        return specs

    # ------------------------------------------------------------------
    # Per-collection vector name helpers
    # ------------------------------------------------------------------

    def _qdrant_name(self, spec_key: str) -> str:
        """Return the actual Qdrant collection name for a spec key.

        Reads ``collection_name`` from the spec JSON; falls back to the file
        stem (spec key) itself so existing deployments keep working even if
        the field is absent.

        Args:
            spec_key: Internal registry key, i.e. the JSON file stem
                      (e.g. ``"entity_registry"``).

        Returns:
            Qdrant collection name (e.g. ``"ci_entity_registry"``).
        """
        return self.registry_specs.get(spec_key, {}).get("collection_name", spec_key)

    def _vector_names(self, collection_name: str) -> Tuple[str, str]:
        """Return ``(en_vector_name, th_vector_name)`` for a collection.

        Derives names by appending ``_en`` / ``_th`` to the collection's
        ``vector_name`` from its spec JSON, e.g.:

          entity_registry    → ("entity_en",    "entity_th")
          predicate_registry → ("predicate_en", "predicate_th")
          label_registry     → ("label_en",     "label_th")
        """
        base = self.registry_specs.get(collection_name, {}).get("vector_name", "entity")
        return base + _EN_SUFFIX, base + _TH_SUFFIX

    def _sparse_vector_name(self, collection_name: str) -> Optional[str]:
        """Return the sparse vector name for a collection, or None if unset."""
        return self.registry_specs.get(collection_name, {}).get("sparse_vector_name")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _build_dual_vector_config(
        self,
        collection_name: str,
    ) -> Dict[str, Any]:
        """Build the Qdrant vectors_config dict for a collection.

        Reads HNSW / datatype / on_disk settings from the collection's spec
        file so each registry keeps its own tuning (e.g. entity_registry can
        have m=24 while a future metadata_registry uses m=16).
        """
        from qdrant_client.http import models as qm

        spec = self.registry_specs.get(collection_name, {})
        vcfg = spec.get("vector_config", {})
        hnsw_raw = vcfg.get("hnsw_config", {})

        vec_params = qm.VectorParams(
            size=vcfg.get("size", VECTOR_SIZE),
            distance=qm.Distance.COSINE,
            hnsw_config=qm.HnswConfigDiff(
                m=hnsw_raw.get("m", 24),
                ef_construct=hnsw_raw.get("ef_construct", 256),
                payload_m=hnsw_raw.get("payload_m", 24),
            ),
            on_disk=vcfg.get("on_disk", False),
            datatype=qm.Datatype.FLOAT32,
        )

        vec_en, vec_th = self._vector_names(collection_name)
        return {vec_en: vec_params, vec_th: vec_params}

    def _ensure_collections_exist(self):
        """Ensure all required collections exist in Qdrant with dual-vector config.

        If a collection already exists with the old single-vector layout
        (e.g. ``"entity"`` only), it is automatically migrated to the new
        ``"entity_en"`` / ``"entity_th"`` layout without data loss.
        """
        collections = list(self.registry_specs.keys())

        for spec_key in collections:
            qdrant_name = self._qdrant_name(spec_key)
            vec_en, vec_th = self._vector_names(spec_key)
            dual_config = self._build_dual_vector_config(spec_key)

            try:
                info = self.qdrant_client.get_collection(qdrant_name)
                existing_vectors = set(info.config.params.vectors.keys())

                if vec_en in existing_vectors and vec_th in existing_vectors:
                    print(f"Collection OK (dual-vector): {qdrant_name}  [{vec_en}, {vec_th}]")
                else:
                    print(
                        f"Collection '{qdrant_name}' has old vector config "
                        f"{existing_vectors} → migrating to [{vec_en}, {vec_th}]…"
                    )
                    self._migrate_collection(qdrant_name, dual_config, vec_en)

            except Exception:
                self.qdrant_client.create_collection(
                    collection_name=qdrant_name,
                    vectors_config=dual_config,
                    on_disk_payload=True,
                )
                print(f"Created collection (dual-vector): {qdrant_name}  [{vec_en}, {vec_th}]")

            self._ensure_payload_indexes(qdrant_name)

    def _migrate_collection(
        self,
        collection_name: str,
        dual_vector_config: Dict,
        vec_en: str,
    ):
        """Migrate a single-vector collection to the dual-vector layout.

        Existing points are scrolled, the collection is recreated with the
        new dual-vector schema, and every point is re-inserted with its old
        vector promoted to ``vec_en``.  The ``_th`` vector is intentionally
        omitted and ``has_thai=False`` marks the point for future backfill.

        Args:
            collection_name:    Qdrant collection to migrate
            dual_vector_config: New vectors_config dict (from _build_dual_vector_config)
            vec_en:             Name of the English vector slot (e.g. ``"entity_en"``)
        """
        from qdrant_client.http import models as qm

        print(f"  Fetching existing points from '{collection_name}'…")
        all_points: List[Any] = []
        offset = None
        while True:
            result, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                with_vectors=True,
                with_payload=True,
                limit=250,
                offset=offset,
            )
            all_points.extend(result)
            if next_offset is None:
                break
            offset = next_offset

        print(f"  Fetched {len(all_points)} points. Recreating collection…")

        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=dual_vector_config,
            on_disk_payload=True,
        )

        if not all_points:
            return

        migrated: List[qm.PointStruct] = []
        for pt in all_points:
            # pt.vector may be a dict (named vectors) or a plain list (legacy unnamed)
            if isinstance(pt.vector, dict):
                # Try the new EN name first, then any existing named vector, then fallback
                old_vec = (
                    pt.vector.get(vec_en)
                    or next(
                        (v for k, v in pt.vector.items() if not k.endswith(_TH_SUFFIX)),
                        None,
                    )
                )
            else:
                old_vec = pt.vector  # plain list

            vectors: Dict[str, Any] = {}
            if old_vec:
                vectors[vec_en] = old_vec
            # _th vector intentionally omitted — will be set on next real upsert

            migrated.append(
                qm.PointStruct(
                    id=pt.id,
                    vector=vectors,
                    payload={**(pt.payload or {}), "has_thai": False},
                )
            )

        batch_size = 100
        for i in range(0, len(migrated), batch_size):
            self._qdrant_upsert_with_retry(
                collection_name=collection_name,
                points=migrated[i : i + batch_size],
            )

        print(f"  Migration complete: {len(migrated)} points re-inserted.")

    def _ensure_payload_indexes(self, collection_name: str):
        """Create payload indexes required for filtered searches."""
        from qdrant_client.http import models as qm

        indexes = [
            ("is_canonical", qm.PayloadSchemaType.BOOL),
            ("has_thai",     qm.PayloadSchemaType.BOOL),
            ("name",         qm.PayloadSchemaType.KEYWORD),
            ("name_en",      qm.PayloadSchemaType.KEYWORD),
            ("label",        qm.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in indexes:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass  # index already exists

    # ------------------------------------------------------------------
    # UUID helpers
    # ------------------------------------------------------------------

    def _generate_uuid(self, name: str) -> str:
        """Generate a deterministic UUID from a name (English canonical form)."""
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        return str(uuid.uuid5(namespace, name))

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string."""
        import os
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch (max 100 per OpenAI call)."""
        import os
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(
                self._get_embeddings_batch_chunk(client, batch)
            )
        return all_embeddings

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get_embeddings_batch_chunk(self, client: Any, batch: List[str]) -> List[List[float]]:
        """Fetch embeddings for a single chunk of up to 100 texts, with retry."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Qdrant query helpers
    # ------------------------------------------------------------------

    def _query_qdrant_canonical(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        """Search for canonical entities above the similarity threshold.

        Always queries using ``entity_en`` so matching is done in English
        embedding space regardless of the language of the incoming entity.

        Args:
            collection_name: Qdrant collection to search
            query_vector: **English** embedding of the incoming entity
            limit: Maximum number of results

        Returns:
            List of dicts with keys: id, score, payload
        """
        try:
            from qdrant_client.http import models as qm

            vec_en, _ = self._vector_names(collection_name)  # e.g. "entity_en"

            search_results = self.qdrant_client.query_points(
                collection_name=self._qdrant_name(collection_name),
                query=query_vector,
                using=vec_en,             # always search in English space
                limit=limit,
                with_payload=True,
                score_threshold=self.similarity_threshold,
                query_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="is_canonical",
                            match=qm.MatchValue(value=True),
                        )
                    ]
                ),
            )

            return [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in search_results.points
            ]

        except Exception as e:
            print(f"Warning: Failed to query Qdrant for canonicals: {e}")
            return []

    def _query_qdrant(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        lang: str = "en",
    ) -> List[Dict[str, Any]]:
        """General-purpose similarity search (non-filtered).

        Args:
            collection_name: Qdrant collection to search
            query_text: Raw text to embed and search
            limit: Max results
            lang: ``"en"`` or ``"th"`` — selects which named vector to search

        Returns:
            List of dicts with keys: id, score, payload
        """
        try:
            query_vector = self._get_embedding(query_text)
            vec_en, vec_th = self._vector_names(collection_name)
            vector_name = vec_en if lang == "en" else vec_th

            search_results = self.qdrant_client.query_points(
                collection_name=self._qdrant_name(collection_name),
                query=query_vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
            )

            return [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in search_results.points
            ]

        except Exception as e:
            print(f"Warning: Failed to query Qdrant: {e}")
            return []

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_fixed(10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _qdrant_upsert_with_retry(self, collection_name: str, points: list) -> None:
        """Upsert points to Qdrant with automatic retry on transient errors."""
        self.qdrant_client.upsert(collection_name=collection_name, points=points)

    def _batch_upsert_entities(
        self,
        entities: List[Dict[str, Any]],
        collection_name: str,
    ) -> Dict[str, str]:
        """Batch-upsert entities to Qdrant using the dual-vector layout.

        For each entity:
          - ``{vector_name}_en`` vector → embedding of ``name_en`` (falls back to ``name``)
          - ``{vector_name}_th`` vector → embedding of ``name`` only when ``name_en``
                                          is present AND differs from ``name``
          - UUID is derived from the English form so the same concept always
            maps to the same point ID.

        Args:
            entities: List of entity dicts.  Recognised keys:
                        name, name_en, label, type, is_canonical, canonical_id,
                        attributes, attributes_en
            collection_name: Target Qdrant collection

        Returns:
            Mapping of original ``name`` → point UUID
        """
        if not entities:
            return {}

        try:
            from qdrant_client.http import models as qm

            # ── Compute which texts need embedding ────────────────────────────
            # We always need the English embedding.
            # We need the Thai embedding only when name_en is present and ≠ name.
            en_texts: List[str] = []
            th_texts: List[Optional[str]] = []  # None = no Thai embed needed

            for entity in entities:
                name = entity["name"]
                name_en = entity.get("name_en")
                en_texts.append(name_en or name)
                has_thai = bool(name_en and name_en.strip() != name.strip())
                th_texts.append(name if has_thai else None)

            # Collect the unique non-None Thai texts so we can batch them
            unique_th = list({t for t in th_texts if t is not None})
            en_embeddings = self._get_embeddings_batch(en_texts)
            th_embedding_map: Dict[str, List[float]] = {}
            if unique_th:
                th_embeds = self._get_embeddings_batch(unique_th)
                th_embedding_map = dict(zip(unique_th, th_embeds))

            # ── Build PointStructs ────────────────────────────────────────────
            vec_en, vec_th = self._vector_names(collection_name)
            points: List[qm.PointStruct] = []
            for entity, en_vec, th_key in zip(entities, en_embeddings, th_texts):
                name = entity["name"]
                name_en = entity.get("name_en")

                # UUID keyed on English label → same real-world concept = same point
                canonical_key = name_en or name
                point_id = self._generate_uuid(canonical_key)

                # Named vectors — names come from the collection's spec
                vectors: Dict[str, List[float]] = {vec_en: en_vec}
                has_thai = th_key is not None
                if has_thai:
                    vectors[vec_th] = th_embedding_map[th_key]

                # Payload
                label = entity.get("label") or entity.get("type", "")
                is_canonical = entity.get("is_canonical", True)
                payload: Dict[str, Any] = {
                    "name": name,
                    "is_canonical": is_canonical,
                    "has_thai": has_thai,
                    "label": label,
                    "type": label,       # backward-compat alias
                }
                if name_en:
                    payload["name_en"] = name_en
                if has_thai:
                    payload["name_th"] = name
                # Canonical entities store their own point_id as canonical_id so
                # every point always has a non-null canonical_id in the payload.
                if is_canonical:
                    payload["canonical_id"] = str(point_id)
                elif entity.get("canonical_id") is not None:
                    payload["canonical_id"] = entity["canonical_id"]
                if entity.get("attributes"):
                    payload["attributes"] = entity["attributes"]
                if entity.get("attributes_en"):
                    payload["attributes_en"] = entity["attributes_en"]

                points.append(
                    qm.PointStruct(id=point_id, vector=vectors, payload=payload)
                )

            self._qdrant_upsert_with_retry(
                collection_name=self._qdrant_name(collection_name),
                points=points,
            )

            # Return name → point_id mapping
            return {
                entity["name"]: self._generate_uuid(entity.get("name_en") or entity["name"])
                for entity in entities
            }

        except Exception as e:
            print(f"Error: Failed to batch upsert to '{collection_name}' after retries: {e}")
            print(f"  Entities: {[e['name'] for e in entities]}")
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
        """Upsert a single entity.  Convenience wrapper around _batch_upsert_entities."""
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

    # ------------------------------------------------------------------
    # LLM helpers  (unchanged logic, same prompts)
    # ------------------------------------------------------------------

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            import os
            from langchain_openai import ChatOpenAI
            from kg_extractor.utils.parser import (
                get_api_key,
                get_groq_api_key,
                get_openrouter_api_key,
            )

            if self.llm_provider == "openai":
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
            elif self.llm_provider == "groq":
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=get_groq_api_key(),
                    base_url="https://api.groq.com/openai/v1",
                )
            elif self.llm_provider == "nvidia":
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=get_api_key(),
                    base_url="https://integrate.api.nvidia.com/v1",
                )
            else:  # openrouter
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=get_openrouter_api_key(),
                    base_url="https://openrouter.ai/api/v1",
                )

            return llm.invoke(prompt).content

        except Exception as e:
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"canonical": null, "canonical_id": null, "synonyms": [], "reasoning": "LLM error"}'

    def _compare_canonical_with_llm(
        self,
        new_entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        entity_type: str,
    ) -> Dict[str, Any]:
        """Use LLM to decide whether the new entity should replace the existing canonical."""
        prompt = self._create_canonical_comparison_prompt(
            new_entity, existing_canonical, entity_type
        )
        response = self._get_llm_response(prompt)
        return self._parse_canonical_winner(response)

    def _compare_predicate_with_llm(
        self,
        new_predicate: str,
        new_predicate_en: Optional[str],
        existing_canonical: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to decide whether two predicates are semantically equivalent."""
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
        """Parse LLM response for predicate equivalence decision."""
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
            new_is_canonical = (
                bool(data.get("new_is_canonical", False)) if are_equiv else True
            )
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
        """Build the LLM prompt for entity identity + canonical selection."""

        def _format_entity(name: str, name_en: Optional[str], attributes: Dict) -> str:
            lines = [f"  Name: {name}"]
            if name_en:
                lines.append(f"  Name (EN): {name_en}")
            if attributes:
                lines.append(
                    f"  Evidence/Attributes: {json.dumps(attributes, ensure_ascii=False)}"
                )
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

        return f"""You are an expert in entity resolution and knowledge graph curation.

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

    def _parse_canonical_winner(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for canonical winner decision."""
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
            new_is_canonical = (
                bool(data.get("new_is_canonical", False)) if are_same else True
            )
            return {
                "are_same_entity": are_same,
                "new_is_canonical": new_is_canonical,
                "reasoning": data.get("reasoning", ""),
            }
        except Exception as e:
            print(f"Warning: Failed to parse canonical winner response: {e}")
            return {
                "are_same_entity": True,
                "new_is_canonical": False,
                "reasoning": "parse error",
            }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Cosine similarity between two vectors."""
        try:
            import numpy as np
            return float(
                np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            )
        except Exception:
            dot = sum(a * b for a, b in zip(vec1, vec2))
            m1 = sum(a * a for a in vec1) ** 0.5
            m2 = sum(b * b for b in vec2) ** 0.5
            return dot / (m1 * m2) if m1 and m2 else 0.0

    # ------------------------------------------------------------------
    # Core refinement pipeline
    # ------------------------------------------------------------------

    def refine_triples(self, triples_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine all triples using Qdrant for entity resolution.

        Bilingual strategy
        ------------------
        When a triple carries ``name_en`` / ``predicate_en``, the **English**
        embedding is used for Qdrant similarity search so entity resolution
        always happens in English space.  Both ``entity_en`` and ``entity_th``
        vectors are stored on the resulting Qdrant point so downstream layers
        can do language-aware lookup without touching the KG structure.

        The KG itself (subjects, predicates, objects) is always populated with
        the English canonical form — Thai is stored only in Qdrant payloads.

        Args:
            triples_data: Raw triples dict (output of the extractor)

        Returns:
            Refined triples dict with canonical English entities
        """
        self._ensure_collections_exist()

        # ── Collect unique entities preserving full metadata ──────────────────
        # Buckets are keyed by spec_key, driven by triple_role in registry JSON.
        entity_spec     = self._role_to_spec_key.get("entity",    "entity_registry")
        predicate_spec  = self._role_to_spec_key.get("predicate", "predicate_registry")
        label_spec      = self._role_to_spec_key.get("label",     "label_registry")

        registry_items: Dict[str, Dict[str, Dict[str, Any]]] = {
            entity_spec:    {},
            predicate_spec: {},
            label_spec:     {},
        }

        for chunk_data in triples_data.get("chunks", []):
            for triple in chunk_data.get("triples", []):
                subj = triple["subject"]
                obj = triple["object"]

                # Subject
                subj_name = subj["name"]
                if subj_name not in registry_items[entity_spec]:
                    registry_items[entity_spec][subj_name] = {
                        "name": subj_name,
                        "name_en": subj.get("name_en"),
                        "label": subj.get("label", ""),
                        "attributes": subj.get("attributes", {}),
                        "attributes_en": subj.get("attributes_en", {}),
                    }

                # Predicate
                pred = triple["predicate"]
                pred_en = triple.get("predicate_en")
                if pred not in registry_items[predicate_spec]:
                    registry_items[predicate_spec][pred] = {
                        "name": pred,
                        "name_en": pred_en,
                        "label": "predicate",
                    }

                # Object
                obj_name = obj["name"]
                if obj_name not in registry_items[entity_spec]:
                    registry_items[entity_spec][obj_name] = {
                        "name": obj_name,
                        "name_en": obj.get("name_en"),
                        "label": obj.get("label", ""),
                        "attributes": obj.get("attributes", {}),
                        "attributes_en": obj.get("attributes_en", {}),
                    }

                # Ontology labels (already English PascalCase)
                for lbl in (subj.get("label", ""), obj.get("label", "")):
                    if lbl and lbl not in registry_items[label_spec]:
                        registry_items[label_spec][lbl] = {
                            "name": lbl,
                            "name_en": lbl,
                            "label": "label",
                        }

        # ── Process each registry ─────────────────────────────────────────────
        entity_cache: Dict[tuple, Dict[str, Any]] = {}
        canonical_change_log: List[Dict[str, Any]] = []

        collections = {
            spec_key: list(items.values())
            for spec_key, items in registry_items.items()
        }

        for collection_name, entity_list in collections.items():
            if not entity_list:
                continue

            print(f"Processing {len(entity_list)} items in {collection_name}…")

            # ── Embed all English forms in one batch ──────────────────────────
            # Resolution always uses entity_en, so we embed name_en (or name).
            en_embed_texts = [e.get("name_en") or e["name"] for e in entity_list]
            print(f"  Getting {len(en_embed_texts)} English embeddings…")
            en_embeddings = self._get_embeddings_batch(en_embed_texts)

            for entity, en_embedding in zip(entity_list, en_embeddings):
                # ── Query canonical registry (English space) ──────────────────
                canonical_matches = self._query_qdrant_canonical(
                    collection_name, en_embedding, limit=1
                )

                if not canonical_matches:
                    # ── New canonical ─────────────────────────────────────────
                    upserted_ids = self._batch_upsert_entities(
                        [{
                            **entity,
                            "is_canonical": True,
                        }],
                        collection_name,
                    )
                    canonical_id = upserted_ids.get(entity["name"])
                    print(f"  New canonical: '{entity['name']}' ({canonical_id})")
                    entity_cache[(collection_name, entity["name"])] = {
                        "original": entity["name"],
                        "canonical": entity.get("name_en") or entity["name"],
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

                    if collection_name == predicate_spec:
                        self._handle_predicate_match(
                            entity, existing_canonical,
                            collection_name, entity_cache, canonical_change_log,
                        )
                    else:
                        self._handle_entity_match(
                            entity, existing_canonical,
                            collection_name, entity_cache, canonical_change_log,
                        )

        # ── Build refined triples (English canonical names in graph) ──────────
        return self._build_refined_output(
            triples_data, entity_cache, canonical_change_log
        )

    # ------------------------------------------------------------------
    # Match-handling sub-routines (reduce refine_triples complexity)
    # ------------------------------------------------------------------

    def _handle_predicate_match(
        self,
        entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        collection_name: str,
        entity_cache: Dict,
        canonical_change_log: List,
    ):
        """Handle a predicate that matched an existing canonical."""
        winner = self._compare_predicate_with_llm(
            new_predicate=entity["name"],
            new_predicate_en=entity.get("name_en"),
            existing_canonical=existing_canonical,
        )

        if not winner["are_equivalent"]:
            # Distinct predicate → own canonical
            upserted_ids = self._batch_upsert_entities(
                [{**entity, "is_canonical": True}],
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
                "canonical": entity.get("name_en") or entity["name"],
                "canonical_id": canonical_id,
                "is_new": True,
                "is_canonical": True,
                "label": "predicate",
                "name_en": entity.get("name_en"),
                "attributes": {},
                "attributes_en": {},
            }

        elif not winner["new_is_canonical"]:
            # Equivalent → existing stays canonical
            upserted_ids = self._batch_upsert_entities(
                [{
                    **entity,
                    "is_canonical": False,
                    "canonical_id": existing_canonical["id"],
                }],
                collection_name,
            )
            canonical_name = (
                existing_canonical["payload"].get("name_en")
                or existing_canonical["payload"].get("name", entity["name"])
            )
            print(
                f"  Predicate '{entity['name']}' merged into canonical '{canonical_name}'"
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
            # Equivalent → new predicate is the better canonical
            canonical_form = (
                winner.get("canonical_form")
                or entity.get("name_en")
                or entity["name"]
            )
            upserted_ids = self._batch_upsert_entities(
                [{**entity, "is_canonical": True}],
                collection_name,
            )
            new_id = upserted_ids.get(entity["name"])
            old_id = existing_canonical["id"]

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
                f"'{existing_canonical['payload'].get('name')}' ({old_id}) "
                f"→ '{entity['name']}' ({new_id})"
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

    def _handle_entity_match(
        self,
        entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        collection_name: str,
        entity_cache: Dict,
        canonical_change_log: List,
    ):
        """Handle an entity / label that matched an existing canonical."""
        winner = self._compare_canonical_with_llm(
            new_entity=entity,
            existing_canonical=existing_canonical,
            entity_type=entity.get("label", collection_name),
        )

        if not winner.get("are_same_entity", True):
            # Distinct entity → new canonical
            upserted_ids = self._batch_upsert_entities(
                [{**entity, "is_canonical": True}],
                collection_name,
            )
            canonical_id = upserted_ids.get(entity["name"])
            print(
                f"  New canonical (distinct from "
                f"'{existing_canonical['payload'].get('name')}'): "
                f"'{entity['name']}' ({canonical_id})"
            )
            entity_cache[(collection_name, entity["name"])] = {
                "original": entity["name"],
                "canonical": entity.get("name_en") or entity["name"],
                "canonical_id": canonical_id,
                "is_new": True,
                "is_canonical": True,
                "label": entity.get("label", ""),
                "name_en": entity.get("name_en"),
                "attributes": entity.get("attributes", {}),
                "attributes_en": entity.get("attributes_en", {}),
            }

        elif not winner["new_is_canonical"]:
            # Same entity → existing stays canonical
            upserted_ids = self._batch_upsert_entities(
                [{
                    **entity,
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
                "canonical": (
                    existing_canonical["payload"].get("name_en")
                    or existing_canonical["payload"].get("name", entity["name"])
                ),
                "canonical_id": existing_canonical["id"],
                "is_new": False,
                "is_canonical": False,
                "label": entity.get("label", ""),
                "name_en": entity.get("name_en"),
                "attributes": entity.get("attributes", {}),
                "attributes_en": entity.get("attributes_en", {}),
            }

        else:
            # Same entity → new entity becomes canonical
            upserted_ids = self._batch_upsert_entities(
                [{**entity, "is_canonical": True}],
                collection_name,
            )
            new_id = upserted_ids.get(entity["name"])
            old_id = existing_canonical["id"]

            try:
                self.qdrant_client.set_payload(
                    collection_name=collection_name,
                    payload={"is_canonical": False, "canonical_id": new_id},
                    points=[old_id],
                )
            except Exception as e:
                print(f"  Warning: Failed to demote old canonical: {e}")

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
                "canonical": entity.get("name_en") or entity["name"],
                "canonical_id": new_id,
                "is_new": True,
                "is_canonical": True,
                "label": entity.get("label", ""),
                "name_en": entity.get("name_en"),
                "attributes": entity.get("attributes", {}),
                "attributes_en": entity.get("attributes_en", {}),
            }

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------

    def _build_refined_output(
        self,
        triples_data: Dict[str, Any],
        entity_cache: Dict[tuple, Dict[str, Any]],
        canonical_change_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the refined triples output dict.

        Canonical names used in the graph are always the **English** form:
        ``refined["canonical"]`` was set to ``name_en or name`` throughout the
        resolution pipeline above.
        """
        refined_chunks = []

        for chunk_data in triples_data.get("chunks", []):
            chunk_id = chunk_data["chunk_id"]
            refined_triples = []

            for triple in chunk_data.get("triples", []):
                subj_raw = triple["subject"]
                obj_raw = triple["object"]

                # Pull from cache; fall back to raw values if a cache miss occurs
                _entity_spec    = self._role_to_spec_key.get("entity",    "entity_registry")
                _predicate_spec = self._role_to_spec_key.get("predicate", "predicate_registry")
                _label_spec     = self._role_to_spec_key.get("label",     "label_registry")

                subject_key = (_entity_spec, subj_raw["name"])
                refined_subject = entity_cache.get(subject_key, {
                    "original": subj_raw["name"],
                    "canonical": subj_raw.get("name_en") or subj_raw["name"],
                    "canonical_id": None,
                    "is_new": True,
                    "label": subj_raw.get("label", ""),
                    "name_en": subj_raw.get("name_en"),
                    "attributes": subj_raw.get("attributes", {}),
                    "attributes_en": subj_raw.get("attributes_en", {}),
                })

                predicate_key = (_predicate_spec, triple["predicate"])
                refined_predicate = entity_cache.get(predicate_key, {
                    "original": triple["predicate"],
                    "canonical": triple.get("predicate_en") or triple["predicate"],
                    "canonical_id": None,
                    "is_new": True,
                    "label": "predicate",
                    "name_en": triple.get("predicate_en"),
                })

                object_key = (_entity_spec, obj_raw["name"])
                refined_object = entity_cache.get(object_key, {
                    "original": obj_raw["name"],
                    "canonical": obj_raw.get("name_en") or obj_raw["name"],
                    "canonical_id": None,
                    "is_new": True,
                    "label": obj_raw.get("label", ""),
                    "name_en": obj_raw.get("name_en"),
                    "attributes": obj_raw.get("attributes", {}),
                    "attributes_en": obj_raw.get("attributes_en", {}),
                })

                obj_label_raw = obj_raw.get("label", "")
                object_label_key = (_label_spec, obj_label_raw)
                refined_object_label = entity_cache.get(object_label_key, {
                    "original": obj_label_raw,
                    "canonical": obj_label_raw,
                    "canonical_id": None,
                    "is_new": True,
                    "label": "label",
                })

                canonical_obj_label = (
                    refined_object_label["canonical"]
                    or refined_object.get("label", "")
                    or obj_label_raw
                )

                # ── Canonical (English) triple ────────────────────────────────
                refined_triple: Dict[str, Any] = {
                    "subject": {
                        # Graph uses English canonical name
                        "name": refined_subject["canonical"],
                        "original_name": subj_raw["name"],
                        "label": refined_subject.get("label", ""),
                    },
                    "predicate": refined_predicate["canonical"],
                    "original_predicate": triple["predicate"],
                    "object": {
                        "name": refined_object["canonical"],
                        "original_name": obj_raw["name"],
                        "label": canonical_obj_label,
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

                # ── Preserve bilingual metadata for downstream use ────────────
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
                    refined_triple["relationship_attributes_en"] = triple[
                        "relationship_attributes_en"
                    ]

                refined_triples.append(refined_triple)

            refined_chunks.append({"chunk_id": chunk_id, "triples": refined_triples})

        return {
            "source_file": triples_data.get("source_file"),
            "total_chunks": len(refined_chunks),
            "total_triples": sum(len(c["triples"]) for c in refined_chunks),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "refinement_applied": True,
            "canonical_change_log": canonical_change_log,
            "chunks": refined_chunks,
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_refined_triples(
        self,
        refined_data: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Save refined triples to a JSON file."""
        from pathlib import Path

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(refined_data, f, indent=2, ensure_ascii=False)

        return str(output_file)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def refine_triples_from_file(
    input_path: str,
    output_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    similarity_threshold: float = 0.85,
) -> str:
    """Refine triples from a JSON file and save the result.

    Args:
        input_path: Path to input triples JSON file
        output_path: Path to save refined triples (default: <input>_refined.json)
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key
        llm_provider: LLM provider for canonical comparison
        llm_model: Model for LLM analysis
        similarity_threshold: Cosine similarity threshold for entity matching

    Returns:
        Path to saved refined triples file
    """
    from pathlib import Path

    with open(input_path, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    refiner = TripleRefiner(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm_provider=llm_provider,
        llm_model=llm_model,
        similarity_threshold=similarity_threshold,
    )

    refined_data = refiner.refine_triples(triples_data)

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_refined{p.suffix}")

    return refiner.save_refined_triples(refined_data, output_path)