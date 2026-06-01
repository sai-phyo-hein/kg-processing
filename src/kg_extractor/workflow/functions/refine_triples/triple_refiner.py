"""Triple refinement module for entity resolution using Qdrant.

Single-vector architecture (entity / predicate / label registries)
------------------------------------------------------------------
Every point in the entity/predicate/label registries carries **one named
dense vector** whose name is derived from the collection's ``vector_name``
field in its registry spec JSON:

  • ``{vector_name}``  – OpenAI embedding of the entity label (English).

Per-collection vector names (from registry spec JSON files):
  entity_registry    → entity    (sparse: entity_vector)
  predicate_registry → predicate (sparse: predicate_vector)
  label_registry     → label     (sparse: label_vector)
  evidence_registry   → evidence  (sparse: evidence_vector)

Evidence collection (dual-vector)
----------------------------------
A separate ``evidence_registry`` collection stores one point per triple.
Each point carries **two** named dense vectors:

  • ``evidence_quote_en``  – text-embedding-3-large embedding of the English evidence quote
  • ``evidence_quote``  – text-embedding-3-large embedding of the Thai evidence quote

Payload fields per evidence point:
  subject        – original English subject name
  predicate      – original English predicate
  object         – original English object name
  community_id   – community id (when present)

All labels, entity names, and predicates are in English.  Qdrant queries
use the single vector per collection for entity resolution.

Point UUID
----------
Derived deterministically from the entity name so that the same real-world
concept always maps to the same Qdrant point.

Payload schema (entity / predicate / label registries)
------------------------------------------------------
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
from kg_extractor.utils.model_setup import OPENAI_EMBEDDING_MODEL, REFINEMENT_PROVIDER, REFINEMENT_MODEL, get_embedding_client, get_reasoning_llm
from kg_extractor.utils.llm_response_parser import parse_json_with_repair
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_SIZE = 1536
_EN_SUFFIX = "_en"
_TH_SUFFIX = "_th"
HIGH_CONFIDENCE_THRESHOLD = 0.95
QUERY_BATCH_SIZE = 50

# Evidence collection
EVIDENCE_EMBEDDING_MODEL = "text-embedding-3-large"
EVIDENCE_VECTOR_DIM = 3072          # text-embedding-3-large native dimension

# ---------------------------------------------------------------------------
# System prompts for LLM
# ---------------------------------------------------------------------------
ENTITY_RESOLUTION_SYSTEM_PROMPT = """You are an expert in entity resolution and knowledge graph curation.

Two entities with high vector similarity have been found. Your task has two steps:

**Step 1 – Identity check:** Decide whether these two entities refer to the *same real-world
concept* (i.e. they are aliases, abbreviations, or surface variants of the same thing).
Set "are_same_entity": true ONLY when both names clearly denote the identical concept.
If one is a specific attribute, metric, sub-component, or related-but-distinct concept of the
other (e.g. "Thai E-Commerce Market" vs "Thai E-Commerce Market Annual Value"), they are
DIFFERENT entities — set "are_same_entity": false.

**Step 2 – Canonical selection (only when are_same_entity is true):** Decide which form should
be the canonical entry. The existing canonical keeps its status UNLESS the new entity is clearly
superior (more specific name, better evidence, more complete information).

**Output Format (JSON):**
```json
{
  "are_same_entity": true_or_false,
  "new_is_canonical": true_or_false,
  "reasoning": "brief explanation"
}
```

Rules:
- If "are_same_entity" is false, set "new_is_canonical" to true (the new entity needs its own canonical entry).
- If "are_same_entity" is true, set "new_is_canonical" based on which form is superior.

Return only valid JSON."""

PREDICATE_RESOLUTION_SYSTEM_PROMPT = """You are an expert in knowledge graph schema design and relationship normalisation.

Two predicates (relationship types) have high vector similarity. Your task:

**Step 1 – Semantic equivalence check:** Decide whether these two predicates express the
*same relationship type* in a knowledge graph. Synonyms and near-synonyms count as
equivalent (e.g. "increase" / "grow" / "rise" / "improve" are all equivalent; but
"increase" and "decrease" are NOT equivalent even though they are related).

**Step 2 – Canonical form selection (only when equivalent):** Choose the most standard,
concise, and generic English verb form as the canonical predicate. Prefer:
- Short, common English verbs over long phrases
- Active voice, base form (infinitive without "to")
- Domain-neutral wording when both options work

**Output Format (JSON):**
```json
{
  "are_equivalent": true_or_false,
  "new_is_canonical": true_or_false,
  "canonical_form": "chosen canonical predicate string",
  "reasoning": "brief explanation"
}
```

Rules:
- If "are_equivalent" is false → set "new_is_canonical" to true so the incoming predicate
  gets its own canonical entry; "canonical_form" should be the incoming predicate.
- If "are_equivalent" is true → set "new_is_canonical" based on which form is superior;
  "canonical_form" must be the chosen canonical string.

Return only valid JSON."""


class TripleRefiner:
    """Refine knowledge graph triples using Qdrant for entity resolution."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm_provider: str = REFINEMENT_PROVIDER,
        llm_model: str = REFINEMENT_MODEL,
        registry_info_dir: Optional[str] = None,
        similarity_threshold: float = 0.95,
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

        # Initialize stateful agents with persistent system prompts
        self._init_stateful_agents()

        self._init_qdrant_client()

    # ------------------------------------------------------------------
    # Stateful agent initialization
    # ------------------------------------------------------------------

    def _init_stateful_agents(self):
        """Initialize separate stateful agents with persistent system prompts.
        
        Creates two completely separate agents:
        - Entity agent: for entity/label resolution
        - Predicate agent: for predicate resolution
        
        Each agent has its own LLM instance and message history.
        System messages persist, user messages are cleared after each call.
        """
        from langchain_core.messages import SystemMessage
        from langchain_core.chat_history import InMemoryChatMessageHistory
        
        # Create separate LLM instances for each agent
        self.entity_agent = {
            'llm': get_reasoning_llm(model=self.llm_model, temperature=0.3),
            'history': InMemoryChatMessageHistory()
        }
        
        self.predicate_agent = {
            'llm': get_reasoning_llm(model=self.llm_model, temperature=0.3),
            'history': InMemoryChatMessageHistory()
        }
        
        # Add persistent system messages to each agent's history
        self.entity_agent['history'].add_message(
            SystemMessage(content=ENTITY_RESOLUTION_SYSTEM_PROMPT)
        )
        self.predicate_agent['history'].add_message(
            SystemMessage(content=PREDICATE_RESOLUTION_SYSTEM_PROMPT)
        )

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
    # Collection management
    # ------------------------------------------------------------------

    def _qdrant_name(self, spec_key: str) -> str:
        """Get the Qdrant collection name for a given spec key.
        
        Args:
            spec_key: The registry spec key (e.g., 'entity_registry')
            
        Returns:
            The Qdrant collection name from the spec's collection_name field
        """
        return self.registry_specs[spec_key]["collection_name"]
    
    def _ensure_collections_exist(self):
        """Ensure all required entity/predicate/label collections exist.

        Qdrant requires a payload index on any field used in a filter.
        'is_canonical' is filtered on every canonical query, so we create a
        bool index for it if one does not already exist.
        
        Evidence collection does not need payload indexes (uses vector comparison only).
        """
        from qdrant_client.http import models as qm

        collections = list(self.registry_specs.keys())
        evidence_spec = self._role_to_spec_key.get("evidence", "evidence_registry")

        for spec_key in collections:
            qdrant_name = self._qdrant_name(spec_key)

            try:
                self.qdrant_client.get_collection(qdrant_name)
                print(f"Collection '{qdrant_name}' already exists.")
            except Exception as e:
                raise e

            # Skip payload indexes for evidence collection (uses vector comparison only)
            if spec_key == evidence_spec:
                print(f"  Skipping payload indexes for evidence collection '{qdrant_name}'")
                continue

            # Ensure payload index on is_canonical exists.
            # create_payload_index is idempotent — safe to call every run.
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=qdrant_name,
                    field_name="is_canonical",
                    field_schema=qm.PayloadSchemaType.BOOL,
                )
                print(f"  Payload index on 'is_canonical' ensured for '{qdrant_name}'.")
            except Exception as e:
                print(f"  Warning: Could not create payload index on 'is_canonical' for '{qdrant_name}': {e}")

            predicate_spec = self._role_to_spec_key.get("predicate", "predicate_registry")
            if spec_key == predicate_spec:
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=qdrant_name,
                        field_name="community_id",
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                    print(f"  Payload index on 'community_id' ensured for '{qdrant_name}'.")
                except Exception as e:
                    print(f"  Warning: Could not create payload index on 'community_id' for '{qdrant_name}': {e}")

    # ------------------------------------------------------------------
    # UUID helpers
    # ------------------------------------------------------------------

    def _generate_uuid(self, name: str) -> str:
        """Generate a deterministic UUID from a name (English canonical form)."""
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        return str(uuid.uuid5(namespace, name))

    def _vector_names(self, spec_key: str) -> Tuple[str, str]:
        """Return the (en_vector_name, th_vector_name) for a registry spec.

        Priority for the English vector name:
          1. vector_name_en explicitly set in spec JSON
          2. {vector_name}_en — for bilingual/dual-vector collections
          3. vector_name base as-is — for single-vector collections (e.g. label)

        A collection is single-vector when bilingual=False/absent and
        vector_name_en is not explicitly set.
        """
        spec = self.registry_specs[spec_key]
        base = spec.get("vector_name", spec_key)
        bilingual = spec.get("bilingual", False)

        if "vector_name_en" in spec:
            vec_en = spec["vector_name_en"]
        elif bilingual:
            vec_en = f"{base}{_EN_SUFFIX}"
        else:
            # Single-vector collection: use base name as stored (e.g. "label" not "label_en")
            vec_en = base

        vec_th = spec.get("vector_name_th", f"{base}{_TH_SUFFIX}")
        return vec_en, vec_th

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string."""
        client = get_embedding_client()
        # Ensure text is not empty to prevent OpenAI API errors
        if not text.strip():
            text = " "
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch (max 100 per OpenAI call)."""
        client = get_embedding_client()
        batch_size = 100
        all_embeddings = []
        # Filter out empty strings to prevent OpenAI API errors
        texts = [t if t.strip() else " " for t in texts]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(
                self._get_embeddings_batch_chunk(client, batch)
            )
        return all_embeddings

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_embeddings_batch_chunk(self, client: Any, batch: List[str]) -> List[List[float]]:
        """Fetch embeddings for a single chunk of up to 100 texts, with retry."""
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
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
        community_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for canonical entities above the similarity threshold.

        Always queries using ``entity_en`` so matching is done in English
        embedding space regardless of the language of the incoming entity.

        Args:
            collection_name: Qdrant collection to search
            query_vector: **English** embedding of the incoming entity
            limit: Maximum number of results
            community_id: When set (predicates only), restricts matches to the
                          same community so cross-community canonicals are never
                          merged together.

        Returns:
            List of dicts with keys: id, score, payload
        """
        try:
            from qdrant_client.http import models as qm

            vec_en, _ = self._vector_names(collection_name)  # e.g. "entity_en"

            must_conditions = [
                qm.FieldCondition(
                    key="is_canonical",
                    match=qm.MatchValue(value=True),
                )
            ]
            if community_id:
                must_conditions.append(
                    qm.FieldCondition(
                        key="community_id",
                        match=qm.MatchValue(value=community_id),
                    )
                )

            search_results = self.qdrant_client.query_points(
                collection_name=self._qdrant_name(collection_name),
                query=query_vector,
                using=vec_en,             # always search in English space
                limit=limit,
                with_payload=True,
                score_threshold=self.similarity_threshold,
                query_filter=qm.Filter(must=must_conditions),
            )

            return [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in search_results.points
            ]

        except Exception as e:
            print(f"Warning: Failed to query Qdrant for canonicals: {e}")
            return []

    def _query_qdrant_canonical_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 1,
        community_ids: Optional[List[Optional[str]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Batch query for canonical entities in chunks.

        Args:
            collection_name: Qdrant collection to search
            query_vectors: List of embedding vectors
            limit: Maximum results per query
            community_ids: Optional per-vector community_id values.  When a
                           community_id is non-None (predicates only), the
                           corresponding query is filtered to that community so
                           cross-community canonicals are never matched.

        Returns:
            Dict mapping query_vector index to list of matches [{"id", "score", "payload"}, ...]
        """
        from qdrant_client.http import models as qm
        
        results: Dict[int, List[Dict[str, Any]]] = {}
        vec_en, _ = self._vector_names(collection_name)

        # Build a per-index filter that optionally adds a community_id condition.
        def _make_filter(idx: int) -> qm.Filter:
            must_conditions = [
                qm.FieldCondition(
                    key="is_canonical",
                    match=qm.MatchValue(value=True),
                )
            ]
            cid = (community_ids[idx] if community_ids and idx < len(community_ids) else None)
            if cid:
                must_conditions.append(
                    qm.FieldCondition(
                        key="community_id",
                        match=qm.MatchValue(value=cid),
                    )
                )
            return qm.Filter(must=must_conditions)

        canonical_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="is_canonical",
                    match=qm.MatchValue(value=True),
                )
            ]
        )

        for i in range(0, len(query_vectors), QUERY_BATCH_SIZE):
            chunk_end = min(i + QUERY_BATCH_SIZE, len(query_vectors))
            chunk = query_vectors[i:chunk_end]

            requests = [
                qm.QueryRequest(
                    query=qm.NearestQuery(nearest=vec),
                    limit=limit,
                    with_payload=True,
                    score_threshold=self.similarity_threshold,
                    filter=_make_filter(i + local_idx),
                    using=vec_en,
                )
                for local_idx, vec in enumerate(chunk)
            ]

            max_attempts = 4
            wait = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    self._init_qdrant_client()
                    batch_results = self.qdrant_client.query_batch_points(
                        collection_name=self._qdrant_name(collection_name),
                        requests=requests,
                    )
                    for idx, query_response in enumerate(batch_results):
                        results[i + idx] = [
                            {"id": r.id, "score": r.score, "payload": r.payload}
                            for r in query_response.points
                        ]
                    break
                except Exception as e:
                    if attempt < max_attempts:
                        print(f"Warning: Query chunk {i // QUERY_BATCH_SIZE + 1} attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait}s...")
                        import time; time.sleep(wait)
                        wait = min(wait * 2, 60)
                    else:
                        print(f"Warning: Query chunk {i // QUERY_BATCH_SIZE + 1} failed after {max_attempts} attempts: {e}. Skipping chunk.")

        return results

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

    def _upsert_chunk_with_retry(self, collection_name: str, chunk: list, chunk_label: str) -> None:
        """Upsert a single chunk with per-chunk retry logic."""
        import time
        max_attempts = 5
        wait = 5
        for attempt in range(1, max_attempts + 1):
            self._init_qdrant_client()
            try:
                self.qdrant_client.upload_points(collection_name=collection_name, points=chunk, wait=True)
                return
            except Exception as e:
                print(f"  ⚠️  {chunk_label} failed (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise
                print(f"  ⏳ Retrying in {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, 60)

    def _qdrant_upsert_with_retry(self, collection_name: str, points: list) -> None:
        """Upsert points to Qdrant in chunks with per-chunk retry.

        wait=True blocks until Qdrant finishes indexing each point before returning.
        Without this, HNSW index updates are async — a query fired immediately after
        will not find the new point, making it appear as a new canonical on every
        re-run of the same data.
        """
        import time
        print(f"Upserting {len(points)} points to collection '{collection_name}'...")
        chunk_size = 50
        chunks = [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]
        for idx, chunk in enumerate(chunks):
            chunk_label = f"Chunk {idx + 1}/{len(chunks)} ({len(chunk)} points)"
            print(f"  Upserting {chunk_label}...")
            self._upsert_chunk_with_retry(collection_name, chunk, chunk_label)
            if idx < len(chunks) - 1:
                time.sleep(10)

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
                        name, name_en, label, type, group, is_canonical, canonical_id,
                        attributes, attributes_en
            collection_name: Target Qdrant collection

        Returns:
            Mapping of original ``name`` → point UUID
        """
        if not entities:
            return {}

        try:
            from qdrant_client.http import models as qm

            label_spec = self._role_to_spec_key.get("label", "label_registry")
            is_label_collection = collection_name == label_spec

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

                # UUID keyed on English label → same real-world concept = same point.
                # For predicates the key also encodes community_id so identical
                # predicate names from different communities get distinct points.
                canonical_key = name_en or name
                community_id  = entity.get("community_id")
                if community_id:
                    canonical_key = f"{canonical_key}::{community_id}"
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
                if entity.get("community_id"):
                    payload["community_id"] = entity["community_id"]
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

            # ── Skip points whose payload is already identical in Qdrant ──────
            qdrant_name = self._qdrant_name(collection_name)
            try:
                existing = self.qdrant_client.retrieve(
                    collection_name=qdrant_name,
                    ids=[p.id for p in points],
                    with_payload=True,
                    with_vectors=False,
                )
                existing_payloads = {r.id: r.payload for r in existing}
            except Exception:
                existing_payloads = {}

            new_points = [
                p for p in points
                if p.id not in existing_payloads or existing_payloads[p.id] != p.payload
            ]

            if not new_points:
                return {
                    entity["name"]: self._generate_uuid(
                        f"{entity.get('name_en') or entity['name']}::{entity['community_id']}"
                        if entity.get("community_id")
                        else (entity.get("name_en") or entity["name"])
                    )
                    for entity in entities
                }

            self._qdrant_upsert_with_retry(
                collection_name=qdrant_name,
                points=new_points,
            )

            # Return name → point_id mapping
            return {
                entity["name"]: self._generate_uuid(
                    f"{entity.get('name_en') or entity['name']}::{entity['community_id']}"
                    if entity.get("community_id")
                    else (entity.get("name_en") or entity["name"])
                )
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
    # LLM helpers
    # ------------------------------------------------------------------

    def _get_llm_response(self, user_message: str, is_predicate: bool = False) -> str:
        """Get response from the appropriate stateful agent.
        
        Selects between entity_agent and predicate_agent based on context.
        The system prompt persists in each agent's history.
        User messages are added, processed, then cleared after each call.
        
        Args:
            user_message: Specific data/question for this call
            is_predicate: If True, use predicate agent; else use entity agent
            
        Returns:
            LLM response text
        """
        try:
            # Select the appropriate agent (separate LLM + history)
            agent = self.predicate_agent if is_predicate else self.entity_agent
            
            # Add user message to agent's history (system message already persists)
            agent['history'].add_user_message(user_message)
            
            # Get all messages (system + user)
            messages = agent['history'].messages
            
            # Invoke the agent's LLM
            response = agent['llm'].invoke(messages)
            
            # Clear user message to prevent accumulation (keep only system message)
            # Remove the last message (user message) and any AI response
            while len(agent['history'].messages) > 1:  # Keep only the first message (system)
                agent['history'].messages.pop()
            
            return response.content
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
        user_message = self._create_canonical_comparison_message(
            new_entity, existing_canonical, entity_type
        )
        response = self._get_llm_response(user_message, is_predicate=False)
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

        user_message = f"""**Existing Canonical:**  {existing_name}
  (original: {old_payload.get("name", "")})
  ID: {existing_canonical.get("id")}
  Similarity score: {existing_canonical.get("score", 0):.3f}

**Incoming Predicate:**  {incoming_name}
  (original: {new_predicate})"""

        response = self._get_llm_response(user_message, is_predicate=True)
        return self._parse_predicate_equivalence(response, incoming_name)

    def _compare_batch_with_llm(
        self,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any], str]],
        is_predicate: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch LLM comparison for multiple entity-canonical pairs.

        Args:
            pairs: List of (entity, existing_canonical, entity_type) tuples
            is_predicate: True for predicates, False for entities/labels

        Returns:
            Dict mapping entity["name"] → decision dict
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        if not pairs:
            return results

        if is_predicate:
            # For predicates: use predicate agent
            for entity, canonical, _ in pairs:
                entity_name = entity["name"]
                old_payload = canonical.get("payload", {})
                existing_name = old_payload.get("name_en") or old_payload.get("name", "")
                incoming_name = entity.get("name_en") or entity["name"]

                user_message = f"""**Existing Canonical:**  {existing_name}
  (original: {old_payload.get("name", "")})
  ID: {canonical.get("id")}
  Similarity score: {canonical.get("score", 0):.3f}

**Incoming Predicate:**  {incoming_name}
  (original: {entity["name"]})"""

                response = self._get_llm_response(user_message, is_predicate=True)
                results[entity_name] = self._parse_predicate_equivalence(response, incoming_name)
        else:
            # For entities/labels: use entity agent
            for entity, canonical, entity_type in pairs:
                entity_name = entity["name"]
                user_message = self._create_canonical_comparison_message(
                    entity, canonical, entity_type
                )
                response = self._get_llm_response(user_message, is_predicate=False)
                results[entity_name] = self._parse_canonical_winner(response)

        return results

    def _parse_predicate_equivalence(
        self,
        response: str,
        fallback_name: str,
    ) -> Dict[str, Any]:
        """Parse LLM response for predicate equivalence decision."""
        try:
            data = parse_json_with_repair(response)
            if not data:
                raise ValueError("Failed to parse JSON")
            
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

    def _create_canonical_comparison_message(
        self,
        new_entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        entity_type: str,
    ) -> str:
        """Build the user message for entity identity + canonical selection.
        
        Returns only the data portion - system instructions are in ENTITY_RESOLUTION_SYSTEM_PROMPT.
        """

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

        return f"""**Entity Type:** {entity_type}

**Existing Canonical (currently registered):**
{old_text}
  ID: {existing_canonical.get("id")}
  Similarity score: {existing_canonical.get("score", 0):.3f}

**New Entity (incoming):**
{new_text}"""

    def _parse_canonical_winner(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for canonical winner decision."""
        try:
            data = parse_json_with_repair(response)
            if not data:
                raise ValueError("Failed to parse JSON")
            
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


    def _dedup_within_batch(
        self,
        entities: List[Dict[str, Any]],
        embeddings: List[List[float]],
        is_predicate: bool = False,
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """Greedy within-batch deduplication using cosine similarity.

        Newly extracted entities are compared against each other *before*
        querying Qdrant.  Without this step, two similar entities that are
        both absent from Qdrant would each be stored as a new canonical,
        missing an intra-batch duplicate.

        Algorithm (greedy, O(n²)):
          Iterate through entities in order.  For each entity, compute cosine
          similarity against every already-chosen representative.  If the best
          match exceeds ``self.similarity_threshold``, the entity becomes an
          alias of that representative; otherwise it becomes a new representative.

        Args:
            entities:    Entity dicts in the current batch.
            embeddings:  Corresponding English embeddings (parallel to entities).
            is_predicate: Reserved for future LLM routing; unused today.

        Returns:
            rep_indices:  Indices (into ``entities``) of cluster representatives.
                          Only these are sent to Qdrant.
            alias_pairs:  List of ``(alias_idx, rep_idx, score)`` tuples for
                          entities that are sufficiently similar to a representative
                          and will inherit its resolved canonical.
        """
        n = len(entities)
        if n <= 1:
            return list(range(n)), []

        rep_indices: List[int] = []
        alias_pairs: List[Tuple[int, int, float]] = []

        for i in range(n):
            best_rep_idx: Optional[int] = None
            best_score = 0.0
            for rep_idx in rep_indices:
                score = self._cosine_similarity(embeddings[i], embeddings[rep_idx])
                if score >= self.similarity_threshold and score > best_score:
                    best_score = score
                    best_rep_idx = rep_idx

            if best_rep_idx is None:
                rep_indices.append(i)
            else:
                alias_pairs.append((i, best_rep_idx, best_score))

        return rep_indices, alias_pairs

    def _process_collection(
        self,
        collection_name: str,
        entity_list: List[Dict[str, Any]],
        is_predicate: bool,
        is_label: bool,
        entity_cache: Dict,
        canonical_change_log: List,
    ) -> None:
        """Embed → within-batch dedup → batch-query → classify → upsert for one registry collection.

        Labels are English-only (single vector).
        Entities and predicates are bilingual (dual vector stored, English queried).

        Progress is printed as a one-line summary per phase rather than per entity.
        """
        kind = "label" if is_label else ("predicate" if is_predicate else "entity")
        n    = len(entity_list)
        print(f"[{collection_name}] Processing {n} {kind}s")

        # ── Step 1: Embed ─────────────────────────────────────────────────────
        # Labels use name directly (already English). Entities/predicates use name_en.
        embed_texts = [
            e["name"] if is_label else (e.get("name_en") or e["name"])
            for e in entity_list
        ]
        en_embeddings = self._get_embeddings_batch(embed_texts)
        print(f"[{collection_name}]   Embedded {n} texts")

        # ── Step 1.5: Within-batch deduplication ─────────────────────────────
        # Compare newly extracted entities against each other so that two
        # similar entities absent from Qdrant don't both become new canonicals.
        rep_indices, alias_pairs = self._dedup_within_batch(
            entity_list, en_embeddings, is_predicate
        )
        rep_entities   = [entity_list[i]   for i in rep_indices]
        rep_embeddings = [en_embeddings[i] for i in rep_indices]
        if alias_pairs:
            alias_names = [entity_list[a]["name"] for a, _, _ in alias_pairs]
            rep_names   = [entity_list[r]["name"] for _, r, _ in alias_pairs]
            print(
                f"[{collection_name}]   Within-batch dedup: "
                f"{len(alias_pairs)} alias(es) → "
                f"{len(rep_entities)} representative(s)"
            )
            for alias_idx, rep_idx, score in alias_pairs:
                print(
                    f"    '{entity_list[alias_idx]['name']}' "
                    f"→ '{entity_list[rep_idx]['name']}' (score={score:.3f})"
                )

        # ── Step 2: Batch query Qdrant (representatives only) ─────────────────
        # For predicates, pass per-representative community_ids so the query
        # filters results to the same community, preventing cross-community merges.
        rep_community_ids: Optional[List[Optional[str]]] = None
        if is_predicate:
            rep_community_ids = [rep_entities[i].get("community_id") for i in range(len(rep_entities))]

        all_matches = self._query_qdrant_canonical_batch(
            collection_name, rep_embeddings, limit=1,
            community_ids=rep_community_ids,
        )
        print(f"[{collection_name}]   Queried Qdrant ({len(all_matches)} matches found)")

        # ── Step 3: Classify each representative ─────────────────────────────
        unmatched     = []          # no Qdrant match → new canonical
        high_conf     = []          # score >= HIGH_CONFIDENCE_THRESHOLD → auto-merge
        uncertain     = []          # ambiguous → LLM decides
        # predicates always go to LLM regardless of score

        for q_idx, entity in enumerate(rep_entities):
            matches = all_matches.get(q_idx, [])
            if not matches:
                unmatched.append(entity)
                continue

            canonical_match = matches[0]
            score = canonical_match.get("score", 0.0)

            if is_predicate or score < HIGH_CONFIDENCE_THRESHOLD:
                uncertain.append((entity, canonical_match))
            else:
                high_conf.append((entity, canonical_match))

        print(
            f"[{collection_name}]   Classified: "
            f"{len(unmatched)} new | "
            f"{len(high_conf)} auto-merge | "
            f"{len(uncertain)} LLM"
        )

        # ── Step 4a: Upsert new canonicals (no Qdrant match) ─────────────────
        if unmatched:
            new_entities = [{**e, "is_canonical": True} for e in unmatched]
            upserted_ids = self._batch_upsert_entities(new_entities, collection_name)
            for entity in unmatched:
                canonical_id   = upserted_ids.get(entity["name"])
                canonical_name = entity.get("name_en") or entity["name"]
                cache_entry    = {
                    "original":     entity["name"],
                    "canonical":    canonical_name,
                    "canonical_id": canonical_id,
                    "is_new":       True,
                    "is_canonical": True,
                }
                if is_label:
                    cache_entry["label"] = "label"
                else:
                    cache_entry.update({
                        "label":         "predicate" if is_predicate else entity.get("label", ""),
                        "name_en":       entity.get("name_en"),
                        "attributes":    {} if is_predicate else entity.get("attributes", {}),
                        "attributes_en": {} if is_predicate else entity.get("attributes_en", {}),
                    })
                if is_predicate:
                    entity_cache[(collection_name, entity["name"], entity.get("community_id"))] = cache_entry
                else:
                    entity_cache[(collection_name, entity["name"])] = cache_entry
            print(f"[{collection_name}]   Upserted {len(unmatched)} new canonicals")

        # ── Step 4b: SKIP auto-merge (merging disabled) ──────────────────────
        if high_conf:
            print(f"[{collection_name}]   Skipped {len(high_conf)} auto-merge candidates (merging disabled)")

        # ── Step 4c: SKIP LLM decisions (merging disabled) ────────────────────
        if uncertain:
            print(f"[{collection_name}]   Skipped {len(uncertain)} LLM decision candidates (merging disabled)")

        # ── Step 4d: Upsert within-batch aliases ──────────────────────────────
        # All representatives have now been processed and are in entity_cache.
        # Aliases inherit the resolved canonical of their representative.
        if alias_pairs:
            alias_points: List[Dict[str, Any]] = []
            for alias_idx, rep_idx, _score in alias_pairs:
                alias_entity = entity_list[alias_idx]
                rep_entity   = entity_list[rep_idx]
                rep_cache_key = (
                    (collection_name, rep_entity["name"], rep_entity.get("community_id"))
                    if is_predicate
                    else (collection_name, rep_entity["name"])
                )
                rep_cache    = entity_cache.get(rep_cache_key)
                if rep_cache:
                    canonical_id   = rep_cache["canonical_id"]
                    canonical_name = rep_cache["canonical"]
                    alias_points.append(
                        {**alias_entity, "is_canonical": False, "canonical_id": canonical_id}
                    )
                    cache_entry: Dict[str, Any] = {
                        "original":     alias_entity["name"],
                        "canonical":    canonical_name,
                        "canonical_id": canonical_id,
                        "is_new":       False,
                        "is_canonical": False,
                    }
                    if is_label:
                        cache_entry["label"] = "label"
                    else:
                        cache_entry.update({
                            "label":         "predicate" if is_predicate else alias_entity.get("label", ""),
                            "name_en":       alias_entity.get("name_en"),
                            "attributes":    {} if is_predicate else alias_entity.get("attributes", {}),
                            "attributes_en": {} if is_predicate else alias_entity.get("attributes_en", {}),
                        })
                    alias_cache_key = (
                        (collection_name, alias_entity["name"], alias_entity.get("community_id"))
                        if is_predicate
                        else (collection_name, alias_entity["name"])
                    )
                    entity_cache[alias_cache_key] = cache_entry
                else:
                    # Unexpected: representative not resolved; fall back to standalone canonical
                    print(
                        f"  Warning: representative '{rep_entity['name']}' not in cache; "
                        f"treating '{alias_entity['name']}' as a new canonical"
                    )
                    fallback_ids = self._batch_upsert_entities(
                        [{**alias_entity, "is_canonical": True}], collection_name
                    )
                    fallback_id   = fallback_ids.get(alias_entity["name"])
                    fallback_name = alias_entity.get("name_en") or alias_entity["name"]
                    fallback_entry: Dict[str, Any] = {
                        "original":     alias_entity["name"],
                        "canonical":    fallback_name,
                        "canonical_id": fallback_id,
                        "is_new":       True,
                        "is_canonical": True,
                    }
                    if is_label:
                        fallback_entry["label"] = "label"
                    else:
                        fallback_entry.update({
                            "label":         "predicate" if is_predicate else alias_entity.get("label", ""),
                            "name_en":       alias_entity.get("name_en"),
                            "attributes":    {} if is_predicate else alias_entity.get("attributes", {}),
                            "attributes_en": {} if is_predicate else alias_entity.get("attributes_en", {}),
                        })
                    fallback_cache_key = (
                        (collection_name, alias_entity["name"], alias_entity.get("community_id"))
                        if is_predicate
                        else (collection_name, alias_entity["name"])
                    )
                    entity_cache[fallback_cache_key] = fallback_entry

            if alias_points:
                self._batch_upsert_entities(alias_points, collection_name)
            print(f"[{collection_name}]   Upserted {len(alias_pairs)} within-batch alias(es)")

    # ------------------------------------------------------------------
    # Evidence collection (dual-vector)
    # ------------------------------------------------------------------

    def _dedup_evidence_within_batch(
        self,
        records: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """Greedy within-batch deduplication for evidence using cosine similarity.

        Compares evidence records based on their English embeddings.  If two evidence
        items have embeddings above the similarity threshold, only one is kept.

        Algorithm (greedy, O(n²)):
          Iterate through evidence records.  For each record, compute cosine
          similarity against every already-chosen representative.  If the best
          match exceeds ``self.similarity_threshold``, the record is considered
          a duplicate; otherwise it becomes a new representative.

        Args:
            records:    Evidence record dicts in the current batch.
            embeddings: Corresponding English embeddings (parallel to records).

        Returns:
            rep_indices:  Indices (into ``records``) of unique representatives.
                          Only these will be upserted to Qdrant.
            dup_pairs:    List of ``(dup_idx, rep_idx, score)`` tuples for
                          records that are duplicates of a representative.
        """
        n = len(records)
        if n <= 1:
            return list(range(n)), []

        rep_indices: List[int] = []
        dup_pairs: List[Tuple[int, int, float]] = []

        for i in range(n):
            best_rep_idx: Optional[int] = None
            best_score = 0.0
            for rep_idx in rep_indices:
                score = self._cosine_similarity(embeddings[i], embeddings[rep_idx])
                if score >= self.similarity_threshold and score > best_score:
                    best_score = score
                    best_rep_idx = rep_idx

            if best_rep_idx is None:
                rep_indices.append(i)
            else:
                dup_pairs.append((i, best_rep_idx, best_score))

        return rep_indices, dup_pairs

    def _get_evidence_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with text-embedding-3-large at EVIDENCE_VECTOR_DIM dimensions.

        Uses a separate OpenAI call so the evidence collection stays independent
        of the entity/predicate embedding model.
        """
        client = get_embedding_client()
        batch_size = 100
        all_embeddings: List[List[float]] = []
        # Filter out empty strings to prevent OpenAI API errors
        texts = [t if t.strip() else " " for t in texts]
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(
                self._get_evidence_embeddings_batch_chunk(client, batch)
            )
        return all_embeddings

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_evidence_embeddings_batch_chunk(
        self, client: Any, batch: List[str]
    ) -> List[List[float]]:
        """Fetch evidence embeddings for a single chunk, with retry."""
        response = client.embeddings.create(
            model=EVIDENCE_EMBEDDING_MODEL,
            input=batch,
            dimensions=EVIDENCE_VECTOR_DIM,
        )
        return [item.embedding for item in response.data]

    def _upsert_evidence_triples(
        self,
        refined_chunks: List[Dict[str, Any]],
    ) -> None:
        """Embed and upsert every refined triple into the evidence collection.

        Each Qdrant point represents one triple.  The English evidence quote is
        embedded into the English vector and the Thai evidence quote into the Thai
        vector (names read from evidence_registry spec), enabling nearest-neighbour
        retrieval by evidence text in either language.
        Subject name, predicate, and object name are stored in the payload.

        The point UUID is derived deterministically from
        ``"<subject>::<predicate>::<object>::<chunk_id>"`` so the same triple is
        never duplicated across re-runs.

        Args:
            refined_chunks: The ``chunks`` list produced by ``_build_refined_output``
                            (each item has ``chunk_id`` and ``triples``).
        """
        from qdrant_client.http import models as qm
        
        # Get vector names from evidence registry spec
        evidence_spec = self._role_to_spec_key.get("evidence", "evidence_registry")
        vec_en, vec_th = self._vector_names(evidence_spec)
        evidence_collection = self._qdrant_name(evidence_spec)

        # ── Collect all triples to embed ──────────────────────────────────────
        records: List[Dict[str, Any]] = []
        for chunk in refined_chunks:
            chunk_id = chunk["chunk_id"]
            for triple in chunk.get("triples", []):
                subj_name = triple["subject"]["name"]
                pred_name = triple["predicate"]
                obj_name  = triple["object"]["name"]
                props     = triple.get("properties", {})

                records.append({
                    "subject":           subj_name,
                    "predicate":         pred_name,
                    "object":            obj_name,
                    "evidence_quote_en": props.get("evidence_quote_en", ""),
                    "evidence_quote":    props.get("evidence_quote", ""),
                    "chunk_id":          chunk_id,
                    "community_id":      props.get("community_id"),
                })

        if not records:
            print("No evidence triples to upsert.")
            return

        print(f"Embedding {len(records)} evidence triples (dual-vector)...")

        # ── Batch-embed English and Thai evidence quotes ──────────────────────
        # Filter out empty strings and replace with placeholder to avoid OpenAI API errors
        en_texts = [r["evidence_quote_en"] if r["evidence_quote_en"].strip() else "[no evidence]" for r in records]
        th_texts = [r["evidence_quote"] if r["evidence_quote"].strip() else "[no evidence]" for r in records]

        en_embeddings = self._get_evidence_embeddings_batch(en_texts)
        th_embeddings = self._get_evidence_embeddings_batch(th_texts)

        # ── Within-batch deduplication ────────────────────────────────────────
        # Compare evidence records against each other based on English embeddings.
        # If two records are similar (above threshold), keep only one.
        rep_indices, dup_pairs = self._dedup_evidence_within_batch(
            records, en_embeddings
        )
        rep_records       = [records[i]       for i in rep_indices]
        rep_en_embeddings = [en_embeddings[i] for i in rep_indices]
        rep_th_embeddings = [th_embeddings[i] for i in rep_indices]

        if dup_pairs:
            dup_info = [
                f"{records[d]['subject']}->{records[d]['predicate']}->{records[d]['object']} "
                f"(sim={score:.3f} to rep {rep_idx})"
                for d, rep_idx, score in dup_pairs[:5]  # Show first 5
            ]
            print(
                f"   Within-batch dedup: {len(dup_pairs)} duplicate(s) removed, "
                f"{len(rep_records)} unique evidence records remain"
            )
            if len(dup_pairs) <= 5:
                for info in dup_info:
                    print(f"      - {info}")
            else:
                for info in dup_info:
                    print(f"      - {info}")
                print(f"      ... and {len(dup_pairs) - 5} more")

        # ── Build PointStructs ────────────────────────────────────────────────
        points: List[qm.PointStruct] = []
        for record, en_vec, th_vec in zip(rep_records, rep_en_embeddings, rep_th_embeddings):
            point_key = (
                f"{record['subject']}::{record['predicate']}"
                f"::{record['object']}::{record['chunk_id']}"
            )
            point_id = self._generate_uuid(point_key)

            payload: Dict[str, Any] = {
                "subject":          record["subject"],
                "predicate":        record["predicate"],
                "object":           record["object"],
                "evidence_quote":   record["evidence_quote"],
                "evidence_quote_en": record["evidence_quote_en"],
            }
            if record["community_id"]:
                payload["community_id"] = record["community_id"]

            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector={
                        vec_en: en_vec,
                        vec_th: th_vec,
                    },
                    payload=payload,
                )
            )

        # ── Skip identical vectors already in Qdrant ──────────────────────────
        try:
            existing = self.qdrant_client.retrieve(
                collection_name=evidence_collection,
                ids=[p.id for p in points],
                with_payload=False,
                with_vectors=True,
            )
            existing_vectors = {r.id: r.vector for r in existing}
        except Exception:
            existing_vectors = {}

        # Compare vectors (both en and th) to determine if update is needed
        import numpy as np
        new_points = []
        for p in points:
            if p.id not in existing_vectors:
                # Point doesn't exist, add it
                new_points.append(p)
            else:
                # Point exists, compare vectors
                existing_vec = existing_vectors[p.id]
                # Check if either vector differs
                en_diff = not np.allclose(existing_vec.get(vec_en, []), p.vector[vec_en], atol=1e-6)
                th_diff = not np.allclose(existing_vec.get(vec_th, []), p.vector[vec_th], atol=1e-6)
                if en_diff or th_diff:
                    new_points.append(p)

        if not new_points:
            print("All evidence vectors already up-to-date in Qdrant.")
            return

        print(f"Upserting {len(new_points)} new/updated evidence points (vector comparison)...")
        self._qdrant_upsert_with_retry(evidence_collection, new_points)
        print(f"✅ Evidence upsert complete ({len(new_points)} points).")

    def refine_triples(self, triples_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine all triples using Qdrant for entity resolution.

        Bilingual strategy
        ------------------
        Entity resolution always happens in English embedding space.
        Entities and predicates store dual vectors (en + th) in Qdrant.
        Labels are English-only (single vector).

        The KG output uses English canonical forms throughout.
        Thai forms are stored only in Qdrant payloads for downstream lookup.
        """
        self._ensure_collections_exist()

        entity_spec    = self._role_to_spec_key.get("entity",    "entity_registry")
        predicate_spec = self._role_to_spec_key.get("predicate", "predicate_registry")
        label_spec     = self._role_to_spec_key.get("label",     "label_registry")

        # ── Collect unique items per registry ─────────────────────────────────
        registry_items: Dict[str, Dict[str, Dict[str, Any]]] = {
            entity_spec:    {},
            predicate_spec: {},
            label_spec:     {},
        }

        for chunk_data in triples_data.get("chunks", []):
            for triple in chunk_data.get("triples", []):
                subj = triple["subject"]
                obj  = triple["object"]

                for name_key, node in [("subject", subj), ("object", obj)]:
                    name = node["name"]
                    if name not in registry_items[entity_spec]:
                        registry_items[entity_spec][name] = {
                            "name":          name,
                            "name_en":       node.get("name_en"),
                            "label":         node.get("label", ""),
                            "attributes":    node.get("attributes", {}),
                            "attributes_en": node.get("attributes_en", {}),
                        }

                pred         = triple["predicate"]
                pred_en      = triple.get("predicate_en")
                community_id = triple.get("properties", {}).get("community_id")
                pred_key     = (pred, community_id)   # composite: same text, diff community → distinct
                if pred_key not in registry_items[predicate_spec]:
                    registry_items[predicate_spec][pred_key] = {
                        "name":         pred,
                        "name_en":      pred_en,
                        "community_id": community_id,
                    }

                for lbl in (subj.get("label", ""), obj.get("label", "")):
                    if lbl and lbl not in registry_items[label_spec]:
                        registry_items[label_spec][lbl] = {"name": lbl}

        total_items = sum(len(v) for v in registry_items.values())
        print(f"Collected {total_items} unique items across {len(registry_items)} registries")

        # ── Embed and upsert evidence triples FIRST (dual-vector) ─────────────
        print("📎 Processing evidence registry first...")
        # Build initial output for evidence (before entity resolution)
        initial_output = {"chunks": triples_data.get("chunks", [])}
        self._upsert_evidence_triples(initial_output.get("chunks", []))
        print("✅ Evidence registry processed\n")

        # ── Process each registry ─────────────────────────────────────────────
        entity_cache:        Dict[tuple, Dict[str, Any]] = {}
        canonical_change_log: List[Dict[str, Any]]       = []

        for collection_name, items in registry_items.items():
            if not items:
                continue
            entity_list  = list(items.values())
            is_pred      = collection_name == predicate_spec
            is_lbl       = collection_name == label_spec
            self._process_collection(
                collection_name, entity_list,
                is_predicate=is_pred,
                is_label=is_lbl,
                entity_cache=entity_cache,
                canonical_change_log=canonical_change_log,
            )
            print(f"[{collection_name}] Cooling down 30s before next collection...")
            import time; time.sleep(30)

        if canonical_change_log:
            print(f"Canonical replacements this run: {len(canonical_change_log)}")

        refined_output = self._build_refined_output(triples_data, entity_cache, canonical_change_log)

        return refined_output

    # ------------------------------------------------------------------
    # Unified match-decision handler
    # ------------------------------------------------------------------

    def _apply_match_decision(
        self,
        entity: Dict[str, Any],
        existing_canonical: Dict[str, Any],
        collection_name: str,
        entity_cache: Dict,
        canonical_change_log: List,
        decision: Dict[str, Any],
        is_predicate: bool = False,
        is_label: bool = False,
    ) -> str:
        """Apply an LLM match decision and update entity_cache + Qdrant.

        Handles all three outcomes uniformly for entities, predicates, and labels:
          - new canonical  (not same / not equivalent)
          - existing stays (same, existing is canonical)
          - replace        (same, incoming is the better canonical)

        Returns a short outcome string for progress tracking:
          "new" | "merged" | "replaced"
        """
        # ── Normalise decision keys across entity vs predicate responses ─────
        # Entity:    are_same_entity  / new_is_canonical
        # Predicate: are_equivalent   / new_is_canonical
        is_same = decision.get("are_same_entity", decision.get("are_equivalent", True))
        new_is_canon = decision.get("new_is_canonical", False)

        old_payload = existing_canonical.get("payload", {})
        existing_name = (
            old_payload.get("name_en") or old_payload.get("name", "")
        )
        incoming_name = entity.get("name_en") or entity["name"]

        # ── Shared cache entry builder ────────────────────────────────────────
        def _cache_entry(
            canonical: str,
            canonical_id: str,
            is_new: bool,
            is_canon: bool,
        ) -> Dict:
            base = {
                "original":     entity["name"],
                "canonical":    canonical,
                "canonical_id": canonical_id,
                "is_new":       is_new,
                "is_canonical": is_canon,
            }
            if is_label:
                base["label"] = "label"
            else:
                base.update({
                    "label":         "predicate" if is_predicate else entity.get("label", ""),
                    "name_en":       entity.get("name_en"),
                    "attributes":    {} if is_predicate else entity.get("attributes", {}),
                    "attributes_en": {} if is_predicate else entity.get("attributes_en", {}),
                })
            return base

        def _cache_key() -> tuple:
            """Return the entity_cache key, community-scoped for predicates."""
            if is_predicate:
                return (collection_name, entity["name"], entity.get("community_id"))
            return (collection_name, entity["name"])

        # ── Outcome A: not the same → new canonical ───────────────────────────
        if not is_same:
            upserted_ids = self._batch_upsert_entities(
                [{**entity, "is_canonical": True}], collection_name
            )
            canonical_id = upserted_ids.get(entity["name"])
            entity_cache[_cache_key()] = _cache_entry(
                incoming_name, canonical_id, is_new=True, is_canon=True
            )
            return "new"

        # ── Outcome B: same, existing stays canonical ─────────────────────────
        if not new_is_canon:
            self._batch_upsert_entities(
                [{**entity, "is_canonical": False, "canonical_id": existing_canonical["id"]}],
                collection_name,
            )
            canonical_id = existing_canonical["payload"].get(
                "canonical_id", existing_canonical["id"]
            )
            entity_cache[_cache_key()] = _cache_entry(
                existing_name,
                canonical_id,
                is_new=False,
                is_canon=False,
            )
            return "merged"

        # ── Outcome C: same, incoming is the better canonical ─────────────────
        canonical_form = (
            decision.get("canonical_form") or incoming_name
        )
        upserted_ids = self._batch_upsert_entities(
            [{**entity, "is_canonical": True}], collection_name
        )
        new_id  = upserted_ids.get(entity["name"])
        old_id  = existing_canonical["id"]

        try:
            self.qdrant_client.set_payload(
                collection_name=self._qdrant_name(collection_name),
                payload={"is_canonical": False, "canonical_id": new_id},
                points=[old_id],
            )
        except Exception as e:
            print(f"  Warning: Failed to demote old canonical '{existing_name}': {e}")

        canonical_change_log.append({
            "collection":         collection_name,
            "old_canonical_id":   old_id,
            "old_canonical_name": old_payload.get("name"),
            "new_canonical_id":   new_id,
            "new_canonical_name": entity["name"],
            "reasoning":          decision.get("reasoning", ""),
        })
        entity_cache[_cache_key()] = _cache_entry(
            canonical_form, new_id, is_new=True, is_canon=True
        )
        return "replaced"

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------
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

                predicate_community_id = triple.get("properties", {}).get("community_id")
                predicate_key = (_predicate_spec, triple["predicate"], predicate_community_id)
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

                canonical_obj_label = refined_object_label["canonical"]

                # ── Canonical (English) triple ────────────────────────────────
                refined_triple: Dict[str, Any] = {
                    "subject": {
                        # Graph uses English canonical name
                        "name": refined_subject["canonical"],
                        "original_name": subj_raw["name"],
                        "label": refined_subject.get("label", ""),
                        # canonical_id is the Qdrant point_id for the resolved canonical entity
                        "canonical_id": refined_subject.get("canonical_id"),
                    },
                    "predicate": refined_predicate["canonical"],
                    "original_predicate": triple["predicate"],
                    "object": {
                        "name": refined_object["canonical"],
                        "original_name": obj_raw["name"],
                        "label": canonical_obj_label,
                        # canonical_id is the Qdrant point_id for the resolved canonical entity
                        "canonical_id": refined_object.get("canonical_id"),
                    },
                    "properties": triple.get("properties", {}),
                    "chunk_id": chunk_id,
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
    llm_provider: str = REFINEMENT_PROVIDER,
    llm_model: str = REFINEMENT_MODEL,
    similarity_threshold: float = 0.95,
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