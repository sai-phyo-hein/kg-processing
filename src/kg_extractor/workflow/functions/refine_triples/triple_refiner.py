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
  evidence_registry  → evidence  (sparse: evidence_vector_en, evidence_vector_th)

Evidence collection (dual-vector, grouped by quote)
----------------------------------------------------
A separate ``evidence_registry`` collection stores one point per unique
evidence quote.  Triples sharing the same ``evidence_quote`` are grouped
and their entities/predicates are combined into the payload lists.

Each point carries **two** named dense vectors:
  • ``evidence_quote_en``  – text-embedding-3-large embedding of the English quote
  • ``evidence_quote``     – text-embedding-3-large embedding of the Thai quote

Payload fields per evidence point:
  entities           – sorted list of unique entity names (subjects + objects)
  predicates         – sorted list of unique predicate names
  evidence_quote     – original source evidence text
  evidence_quote_en  – English translation of the evidence text
  community_id       – community id (when present)

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

Optimisation notes
------------------
  • LLM calls are fired concurrently via ThreadPoolExecutor.
  • _get_llm_response is stateless (no shared history mutation) so it is
    safe to call from multiple threads simultaneously.
  • All upsert decisions within a collection are collected first, then
    flushed in at most three bulk calls (new canonicals / aliases /
    canonical replacements).
  • The redundant ``retrieve`` round-trip before every upsert has been
    removed; upload_points with a deterministic UUID is idempotent.
  • Inter-collection and inter-chunk sleeps have been removed; the Qdrant
    client is only re-initialised when a genuine connection error occurs.
"""

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from kg_extractor.utils.model_setup import (
    EVIDENCE_EMBEDDING_MODEL,
    EVIDENCE_VECTOR_DIM,
    OPENAI_EMBEDDING_MODEL,
    REFINEMENT_MODEL,
    REFINEMENT_PROVIDER,
    get_embedding_client,
    get_reasoning_llm,
)
from kg_extractor.utils.llm_response_parser import parse_json_with_repair
from kg_extractor.utils.sparse_vectors import (
    build_vocab as _build_vocab_shared,
    compute_sparse_tf_batch as _compute_sparse_tf_batch_shared,
    tokenize_sparse,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTOR_SIZE = 1536
_EN_SUFFIX = "_en"
_TH_SUFFIX = "_th"
HIGH_CONFIDENCE_THRESHOLD = 0.95
LLM_REFINEMENT_THRESHOLD = 0.65
QUERY_BATCH_SIZE = 50
# How many pairs to send per LLM batch call.  Larger = fewer calls but bigger
# prompts; 50 is a safe ceiling before context starts hurting quality.
LLM_BATCH_SIZE = 50
# Upsert chunk sizes
EVIDENCE_UPSERT_CHUNK_SIZE = 10
DEFAULT_UPSERT_CHUNK_SIZE = 50

# ---------------------------------------------------------------------------
# System prompts  (batch versions — N pairs per call, one JSON array back)
# ---------------------------------------------------------------------------
ENTITY_RESOLUTION_SYSTEM_PROMPT = """You are an expert in entity resolution and knowledge graph curation.

You will receive a JSON array of entity pairs, each with an "id" and two entity descriptions.
For EVERY pair, decide:

**Step 1 – Identity check:** Do both names refer to the *same real-world concept*?
Set "are_same_entity": true ONLY when they are aliases, abbreviations, or surface variants of
the same thing.  If one is a sub-component, related metric, or distinct concept of the other,
set it to false.

**Disambiguation rules — apply strictly:**

1. **Different semantic labels → NOT the same.** Each entity carries a "label" (PascalCase
   ontology class). Different labels indicate different real-world categories. Do NOT merge
   entities with different labels unless they are clearly the same thing under different
   class names.

2. **Cause/source ≠ effect.** If one entity produces, enables, or is derived from the other,
   they are distinct — even when they share keywords or appear in the same domain.

3. **Whole ≠ part.** An umbrella/compound concept and one of its constituents are distinct.

4. **Entity ≠ process involving that entity.** A thing and the activity or method that
   operates on it are distinct concepts.

When in doubt, default to "are_same_entity": false. Merging distinct entities is worse than
having separate entries that happen to be related.

**Step 2 – Canonical selection (only when are_same_entity is true):** Which form should be
canonical? The existing canonical keeps its status UNLESS the incoming entity is clearly
superior (more specific name, better evidence, more complete information).

Return a JSON array — one object per input pair, in the same order, each with:
  { "id": <same id from input>, "are_same_entity": bool, "new_is_canonical": bool, "reasoning": "brief" }

Rules:
- If "are_same_entity" is false → set "new_is_canonical": true.
- If "are_same_entity" is true  → set "new_is_canonical" based on which form is superior.
- Output ONLY the JSON array.  No prose before or after it."""

PREDICATE_RESOLUTION_SYSTEM_PROMPT = """You are an expert in knowledge graph schema design and relationship normalisation.

You will receive a JSON array of predicate pairs, each with an "id" and two predicate strings.
For EVERY pair, decide:

**Step 1 – Semantic equivalence:** Do the two predicates express the *same relationship type*?
Synonyms and near-synonyms count as equivalent ("increase"/"grow"/"rise" are equivalent;
"increase" and "decrease" are NOT, even though they are related).

**Step 2 – Canonical form (only when equivalent):** Choose the most standard, concise,
domain-neutral English verb form in ALL_CAPS_SNAKE_CASE.

Return a JSON array — one object per input pair, in the same order, each with:
  { "id": <same id from input>, "are_equivalent": bool, "new_is_canonical": bool,
    "canonical_form": "chosen string", "reasoning": "brief" }

Rules:
- If "are_equivalent" is false → "new_is_canonical": true, "canonical_form": incoming predicate.
- If "are_equivalent" is true  → "new_is_canonical" based on which form is superior.
- Output ONLY the JSON array.  No prose before or after it."""


# ---------------------------------------------------------------------------
# TripleRefiner
# ---------------------------------------------------------------------------

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

        self._init_llm_agents()
        self._init_qdrant_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_llm_agents(self) -> None:
        """Create two LLM instances (entity + predicate) with their system prompts.

        The system prompts are stored as plain strings rather than being held
        in a mutable InMemoryChatMessageHistory so that _get_llm_response can
        be called safely from multiple threads simultaneously.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        self._entity_system_msg    = SystemMessage(content=ENTITY_RESOLUTION_SYSTEM_PROMPT)
        self._predicate_system_msg = SystemMessage(content=PREDICATE_RESOLUTION_SYSTEM_PROMPT)
        self._entity_llm    = get_reasoning_llm(model=self.llm_model, temperature=0.3)
        self._predicate_llm = get_reasoning_llm(model=self.llm_model, temperature=0.3)
        # Keep HumanMessage available for type hints in _get_llm_response
        self._HumanMessage = HumanMessage

    def _init_qdrant_client(self) -> None:
        """Initialise (or re-initialise) the Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=300,
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
        """Load registry specifications from the registry_info directory."""
        specs: Dict[str, Dict[str, Any]] = {}
        self._role_to_spec_key: Dict[str, str] = {}

        spec_files = sorted(self.registry_info_dir.glob("*.json"))
        if not spec_files:
            print(f"Warning: No registry spec files found in {self.registry_info_dir}")

        for spec_file in spec_files:
            spec_key = spec_file.stem
            with open(spec_file, "r") as f:
                spec = json.load(f)
            spec.setdefault("collection_name", spec_key)
            specs[spec_key] = spec

            role = spec.get("triple_role")
            if role:
                self._role_to_spec_key[role] = spec_key
                print(f"Registry '{spec_key}' → role '{role}' → Qdrant '{spec['collection_name']}'")

        return specs

    def _qdrant_name(self, spec_key: str) -> str:
        return self.registry_specs[spec_key]["collection_name"]

    def _vector_names(self, spec_key: str) -> Tuple[str, str]:
        """Return (en_vector_name, th_vector_name) for a registry spec."""
        spec    = self.registry_specs[spec_key]
        base    = spec.get("vector_name", spec_key)
        bilingual = spec.get("bilingual", False)

        if "vector_name_en" in spec:
            vec_en = spec["vector_name_en"]
        elif bilingual:
            vec_en = f"{base}{_EN_SUFFIX}"
        else:
            vec_en = base

        vec_th = spec.get("vector_name_th", f"{base}{_TH_SUFFIX}")
        return vec_en, vec_th

    def _sparse_vector_names(self, spec_key: str) -> Tuple[str, str]:
        """Return (sparse_en_name, sparse_th_name) for a registry spec."""
        spec = self.registry_specs[spec_key]
        sparse_th = spec.get("sparse_vector_name_th", "")
        if sparse_th:
            sparse_en = spec.get("sparse_vector_name_en", f"{sparse_th}{_EN_SUFFIX}")
            return sparse_en, sparse_th
        sparse_base = spec.get("sparse_vector_name", "")
        if not sparse_base:
            return "", ""
        return f"{sparse_base}{_EN_SUFFIX}", sparse_base

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collections_exist(self) -> None:
        """Ensure all required collections exist and have the right payload indexes."""
        from qdrant_client.http import models as qm

        evidence_spec = self._role_to_spec_key.get("evidence_quote", "evidence_registry")
        predicate_spec = self._role_to_spec_key.get("predicate", "predicate_registry")

        for spec_key in self.registry_specs:
            qdrant_name = self._qdrant_name(spec_key)
            try:
                self.qdrant_client.get_collection(qdrant_name)
                print(f"Collection '{qdrant_name}' already exists.")
            except Exception as e:
                raise e

            if spec_key == evidence_spec:
                print(f"  Skipping payload indexes for evidence collection '{qdrant_name}'")
                continue

            try:
                self.qdrant_client.create_payload_index(
                    collection_name=qdrant_name,
                    field_name="is_canonical",
                    field_schema=qm.PayloadSchemaType.BOOL,
                )
                print(f"  Payload index on 'is_canonical' ensured for '{qdrant_name}'.")
            except Exception as e:
                print(f"  Warning: Could not create payload index on 'is_canonical' for '{qdrant_name}': {e}")

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
        """Deterministic UUID from an English canonical name."""
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        return str(uuid.uuid5(namespace, name))

    def _entity_uuid(self, entity: Dict[str, Any]) -> str:
        canonical_key = entity.get("name_en") or entity["name"]
        community_id  = entity.get("community_id")
        if community_id:
            canonical_key = f"{canonical_key}::{community_id}"
        return self._generate_uuid(canonical_key)

    # ------------------------------------------------------------------
    # Sparse vector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_sparse(text: str) -> List[str]:
        return tokenize_sparse(text)

    def _build_vocab(self, all_texts: List[str]) -> Dict[str, int]:
        vocab = _build_vocab_shared(all_texts)
        print(f"  Sparse vocab size: {len(vocab)} unique tokens across {len(all_texts)} documents")
        return vocab

    def _compute_sparse_tf_batch(
        self, texts: List[str], vocab: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        return _compute_sparse_tf_batch_shared(texts, vocab)

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
        client = get_embedding_client()
        if not text.strip():
            text = " "
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text)
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str], _attempt: int = 1) -> List[List[float]]:
        """Embed texts in batches of 100 (OpenAI limit), with outer retry."""
        client = get_embedding_client()
        texts  = [t if t.strip() else " " for t in texts]
        result = []
        try:
            for i in range(0, len(texts), 100):
                result.extend(self._get_embeddings_batch_chunk(client, texts[i:i + 100]))
                if i + 100 < len(texts):
                    time.sleep(1)  # inter-batch delay to avoid rate limits
            return result
        except Exception as e:
            if _attempt < 5:
                wait = min(2 ** _attempt * 5, 60)
                print(f"  ⚠️  Embedding batch failed (attempt {_attempt}/5): {e}")
                print(f"     Retrying in {wait}s...")
                time.sleep(wait)
                return self._get_embeddings_batch(texts, _attempt=_attempt + 1)
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_embeddings_batch_chunk(self, client: Any, batch: List[str]) -> List[List[float]]:
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        return [item.embedding for item in response.data]

    def _get_evidence_embeddings_batch(self, texts: List[str], _attempt: int = 1) -> List[List[float]]:
        """Embed evidence texts with the larger evidence model, with outer retry."""
        client = get_embedding_client()
        texts  = [t if t.strip() else " " for t in texts]
        result = []
        try:
            for i in range(0, len(texts), 100):
                result.extend(self._get_evidence_embeddings_batch_chunk(client, texts[i:i + 100]))
                if i + 100 < len(texts):
                    time.sleep(1)  # inter-batch delay to avoid rate limits
            return result
        except Exception as e:
            if _attempt < 5:
                wait = min(2 ** _attempt * 5, 60)
                print(f"  ⚠️  Evidence embedding batch failed (attempt {_attempt}/5): {e}")
                print(f"     Retrying in {wait}s...")
                time.sleep(wait)
                return self._get_evidence_embeddings_batch(texts, _attempt=_attempt + 1)
            raise

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_evidence_embeddings_batch_chunk(
        self, client: Any, batch: List[str]
    ) -> List[List[float]]:
        response = client.embeddings.create(
            model=EVIDENCE_EMBEDDING_MODEL,
            input=batch,
            dimensions=EVIDENCE_VECTOR_DIM,
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # Qdrant query helpers
    # ------------------------------------------------------------------

    def _query_qdrant_canonical_batch(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 1,
        community_ids: Optional[List[Optional[str]]] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Batch vector search against the canonical index.

        Returns a dict mapping query index → list of matches.
        """
        from qdrant_client.http import models as qm

        vec_en, _ = self._vector_names(collection_name)
        threshold = score_threshold if score_threshold is not None else self.similarity_threshold
        results: Dict[int, List[Dict[str, Any]]] = {}

        def _make_filter(idx: int) -> qm.Filter:
            must = [qm.FieldCondition(key="is_canonical", match=qm.MatchValue(value=True))]
            cid  = community_ids[idx] if community_ids and idx < len(community_ids) else None
            if cid:
                must.append(qm.FieldCondition(key="community_id", match=qm.MatchValue(value=cid)))
            return qm.Filter(must=must)

        for chunk_start in range(0, len(query_vectors), QUERY_BATCH_SIZE):
            chunk_end = min(chunk_start + QUERY_BATCH_SIZE, len(query_vectors))
            chunk     = query_vectors[chunk_start:chunk_end]

            requests = [
                qm.QueryRequest(
                    query=qm.NearestQuery(nearest=vec),
                    limit=limit,
                    with_payload=True,
                    score_threshold=threshold,
                    filter=_make_filter(chunk_start + local_idx),
                    using=vec_en,
                )
                for local_idx, vec in enumerate(chunk)
            ]

            max_attempts, wait_s = 4, 5
            for attempt in range(1, max_attempts + 1):
                try:
                    batch_results = self.qdrant_client.query_batch_points(
                        collection_name=self._qdrant_name(collection_name),
                        requests=requests,
                    )
                    for local_idx, qr in enumerate(batch_results):
                        results[chunk_start + local_idx] = [
                            {"id": r.id, "score": r.score, "payload": r.payload}
                            for r in qr.points
                        ]
                    break
                except Exception as e:
                    if attempt < max_attempts:
                        import time
                        print(
                            f"Warning: Query chunk {chunk_start // QUERY_BATCH_SIZE + 1} "
                            f"attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait_s}s..."
                        )
                        time.sleep(wait_s)
                        wait_s = min(wait_s * 2, 60)
                        # Only re-init client on connection-level errors
                        if "connection" in str(e).lower() or "timeout" in str(e).lower():
                            self._init_qdrant_client()
                    else:
                        print(
                            f"Warning: Query chunk {chunk_start // QUERY_BATCH_SIZE + 1} "
                            f"failed after {max_attempts} attempts: {e}. Skipping."
                        )

        return results

    def _query_qdrant(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        lang: str = "en",
    ) -> List[Dict[str, Any]]:
        """General-purpose non-filtered similarity search."""
        try:
            query_vector        = self._get_embedding(query_text)
            vec_en, vec_th      = self._vector_names(collection_name)
            vector_name         = vec_en if lang == "en" else vec_th
            search_results      = self.qdrant_client.query_points(
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

    def _upsert_chunk_with_retry(
        self,
        collection_name: str,
        chunk: list,
        chunk_label: str,
        timeout: int = 300,
    ) -> None:
        import time
        max_attempts, wait_s = 5, 5
        for attempt in range(1, max_attempts + 1):
            try:
                self.qdrant_client.upload_points(
                    collection_name=collection_name,
                    points=chunk,
                    wait=True,
                )
                return
            except Exception as e:
                print(f"  ⚠️  {chunk_label} failed (attempt {attempt}/{max_attempts}): {e}")
                if attempt == max_attempts:
                    raise
                print(f"  ⏳ Retrying in {wait_s}s...")
                time.sleep(wait_s)
                wait_s = min(wait_s * 2, 60)
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    self._init_qdrant_client()

    def _qdrant_upsert_with_retry(
        self,
        collection_name: str,
        points: list,
        chunk_size: Optional[int] = None,
    ) -> None:
        """Upsert points in chunks.  wait=True ensures HNSW is updated before returning."""
        _chunk_size = chunk_size or DEFAULT_UPSERT_CHUNK_SIZE
        print(f"Upserting {len(points)} points to '{collection_name}' (chunk={_chunk_size})...")
        chunks = [points[i:i + _chunk_size] for i in range(0, len(points), _chunk_size)]
        for idx, chunk in enumerate(chunks):
            chunk_label = f"Chunk {idx + 1}/{len(chunks)} ({len(chunk)} points)"
            print(f"  Upserting {chunk_label}...")
            self._upsert_chunk_with_retry(collection_name, chunk, chunk_label)

    def _build_points(
        self,
        entities: List[Dict[str, Any]],
        collection_name: str,
        en_embeddings: List[List[float]],
        th_embedding_map: Dict[str, List[float]],
        th_texts: List[Optional[str]],
    ) -> List[Any]:
        """Build PointStructs from entities + pre-computed embeddings."""
        from qdrant_client.http import models as qm

        vec_en, vec_th = self._vector_names(collection_name)
        points = []
        for entity, en_vec, th_key in zip(entities, en_embeddings, th_texts):
            name         = entity["name"]
            name_en      = entity.get("name_en")
            is_canonical = entity.get("is_canonical", True)
            label        = entity.get("label") or entity.get("type", "")
            point_id     = self._entity_uuid(entity)

            vectors: Dict[str, List[float]] = {vec_en: en_vec}
            has_thai = th_key is not None
            if has_thai:
                vectors[vec_th] = th_embedding_map[th_key]

            payload: Dict[str, Any] = {
                "name":         name,
                "is_canonical": is_canonical,
                "has_thai":     has_thai,
                "label":        label,
                "type":         label,
            }
            if entity.get("community_id"):
                payload["community_id"] = entity["community_id"]
            if name_en:
                payload["name_en"] = name_en
            if has_thai:
                payload["name_th"] = name
            if is_canonical:
                payload["canonical_id"] = str(point_id)
            elif entity.get("canonical_id") is not None:
                payload["canonical_id"] = entity["canonical_id"]
            if entity.get("attributes"):
                payload["attributes"] = entity["attributes"]
            if entity.get("attributes_en"):
                payload["attributes_en"] = entity["attributes_en"]

            points.append(qm.PointStruct(id=point_id, vector=vectors, payload=payload))

        return points

    def _batch_upsert_entities(
        self,
        entities: List[Dict[str, Any]],
        collection_name: str,
    ) -> Dict[str, str]:
        """Embed + upsert a list of entities.

        Returns a mapping of entity["name"] → point UUID.
        The retrieve-before-upsert check has been removed: upload_points
        with a deterministic UUID is already idempotent (same ID = overwrite).
        """
        if not entities:
            return {}

        try:
            # ── Determine which texts need embedding ──────────────────────────
            en_texts: List[str] = []
            th_texts: List[Optional[str]] = []
            for entity in entities:
                name    = entity["name"]
                name_en = entity.get("name_en")
                en_texts.append(name_en or name)
                has_thai = bool(name_en and name_en.strip() != name.strip())
                th_texts.append(name if has_thai else None)

            unique_th      = list({t for t in th_texts if t is not None})
            en_embeddings  = self._get_embeddings_batch(en_texts)
            th_embedding_map: Dict[str, List[float]] = {}
            if unique_th:
                th_embeds        = self._get_embeddings_batch(unique_th)
                th_embedding_map = dict(zip(unique_th, th_embeds))

            points = self._build_points(
                entities, collection_name, en_embeddings, th_embedding_map, th_texts
            )
            self._qdrant_upsert_with_retry(
                collection_name=self._qdrant_name(collection_name),
                points=points,
            )

            return {e["name"]: self._entity_uuid(e) for e in entities}

        except Exception as e:
            print(f"Error: Failed to batch upsert to '{collection_name}': {e}")
            print(f"  Entities: {[e['name'] for e in entities]}")
            raise

    # ------------------------------------------------------------------
    # LLM helpers  (thread-safe: no shared mutable state)
    # ------------------------------------------------------------------

    def _get_llm_response(self, user_message: str, is_predicate: bool = False) -> str:
        """Thread-safe LLM call.

        Builds a fresh [system, human] message list on every call so multiple
        threads can invoke this simultaneously without racing on shared history.
        """
        try:
            system_msg = self._predicate_system_msg if is_predicate else self._entity_system_msg
            llm        = self._predicate_llm        if is_predicate else self._entity_llm
            messages   = [system_msg, self._HumanMessage(content=user_message)]
            response   = llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"canonical": null, "canonical_id": null, "synonyms": [], "reasoning": "LLM error"}'

    def _parse_entity_decision(self, response: str) -> Dict[str, Any]:
        try:
            data     = parse_json_with_repair(response)
            if not data:
                raise ValueError("empty parse")
            are_same = bool(data.get("are_same_entity", True))
            return {
                "are_same_entity":  are_same,
                "new_is_canonical": bool(data.get("new_is_canonical", False)) if are_same else True,
                "reasoning":        data.get("reasoning", ""),
            }
        except Exception as e:
            print(f"Warning: Failed to parse entity decision: {e}")
            return {"are_same_entity": True, "new_is_canonical": False, "reasoning": "parse error"}

    def _parse_predicate_decision(self, response: str, fallback_name: str) -> Dict[str, Any]:
        try:
            data     = parse_json_with_repair(response)
            if not data:
                raise ValueError("empty parse")
            are_eq   = bool(data.get("are_equivalent", True))
            return {
                "are_equivalent":   are_eq,
                "new_is_canonical": bool(data.get("new_is_canonical", False)) if are_eq else True,
                "canonical_form":   data.get("canonical_form", fallback_name),
                "reasoning":        data.get("reasoning", ""),
            }
        except Exception as e:
            print(f"Warning: Failed to parse predicate decision: {e}")
            return {
                "are_equivalent":   True,
                "new_is_canonical": False,
                "canonical_form":   fallback_name,
                "reasoning":        "parse error",
            }

    def _run_llm_batch(
        self,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        is_predicate: bool,
        entity_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve all pairs using the LLM.

        Pairs are split into chunks of LLM_BATCH_SIZE.  Each chunk is sent as
        a single LLM call (N pairs → 1 message → JSON array response).
        All chunks are fired concurrently via ThreadPoolExecutor, so the total
        wall-clock time is roughly one LLM round-trip regardless of how many
        pairs there are (up to rate-limit ceilings).

        Args:
            pairs:        List of (entity, existing_canonical) tuples.
            is_predicate: Route to predicate agent when True.
            entity_type:  Human-readable type string for entity messages.

        Returns:
            Dict mapping entity["name"] → decision dict.
        """
        if not pairs:
            return {}

        # ── Build chunks ──────────────────────────────────────────────────────
        chunks: List[List[Tuple[Dict, Dict]]] = [
            pairs[i:i + LLM_BATCH_SIZE]
            for i in range(0, len(pairs), LLM_BATCH_SIZE)
        ]
        n_chunks = len(chunks)
        print(
            f"  LLM batch: {len(pairs)} pair(s) → {n_chunks} chunk(s) "
            f"of ≤{LLM_BATCH_SIZE}, firing concurrently"
        )

        def _call_chunk(
            chunk: List[Tuple[Dict, Dict]]
        ) -> List[Tuple[str, Dict[str, Any]]]:
            """Send one chunk as a single LLM call; return list of (name, decision)."""
            # Assign stable integer ids so the LLM can reference them
            items = [
                {"id": i, "entity": entity, "canonical": canonical}
                for i, (entity, canonical) in enumerate(chunk)
            ]

            if is_predicate:
                user_msg = self._make_predicate_batch_message(items)
            else:
                user_msg = self._make_entity_batch_message(items, entity_type)

            response = self._get_llm_response(user_msg, is_predicate=is_predicate)
            raw_decisions = self._parse_batch_response(response)

            # Map id → decision, falling back to safe defaults for missing ids
            id_to_decision: Dict[int, Dict] = {d["id"]: d for d in raw_decisions if "id" in d}

            output: List[Tuple[str, Dict[str, Any]]] = []
            for item in items:
                entity      = item["entity"]
                entity_name = entity["name"]
                raw         = id_to_decision.get(item["id"], {})
                if is_predicate:
                    decision = self._parse_predicate_decision(
                        json.dumps(raw), entity.get("name_en") or entity_name
                    )
                else:
                    decision = self._parse_entity_decision(json.dumps(raw))
                output.append((entity_name, decision))
            return output

        # ── Fire all chunks concurrently ──────────────────────────────────────
        results: Dict[str, Dict[str, Any]] = {}
        # Number of workers == number of chunks: every chunk runs in parallel.
        with ThreadPoolExecutor(max_workers=n_chunks) as executor:
            futures = {executor.submit(_call_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    for entity_name, decision in future.result():
                        results[entity_name] = decision
                except Exception as e:
                    # Whole chunk failed — apply safe defaults for every pair in it
                    print(f"Warning: LLM chunk failed: {e}")
                    for entity, _ in chunk:
                        results[entity["name"]] = (
                            {
                                "are_equivalent":   True,
                                "new_is_canonical": False,
                                "canonical_form":   entity.get("name_en") or entity["name"],
                                "reasoning":        "LLM chunk error",
                            }
                            if is_predicate
                            else {
                                "are_same_entity":  True,
                                "new_is_canonical": False,
                                "reasoning":        "LLM chunk error",
                            }
                        )

        return results

    # ------------------------------------------------------------------
    # Batch message builders
    # ------------------------------------------------------------------

    def _make_entity_batch_message(
        self,
        items: List[Dict[str, Any]],
        entity_type: str,
    ) -> str:
        """Build a single user message containing all entity pairs as a JSON array."""
        payload = []
        for item in items:
            entity      = item["entity"]
            canonical   = item["canonical"]
            old_payload = canonical.get("payload", {})
            payload.append({
                "id": item["id"],
                "entity_type": entity_type,
                "existing": {
                    "name":       old_payload.get("name", ""),
                    "name_en":    old_payload.get("name_en"),
                    "label":      old_payload.get("label") or old_payload.get("type", ""),
                    "attributes": old_payload.get("attributes") or {},
                    "id":         canonical.get("id"),
                    "score":      round(canonical.get("score", 0.0), 3),
                },
                "incoming": {
                    "name":       entity.get("name", ""),
                    "name_en":    entity.get("name_en"),
                    "label":      entity.get("label", ""),
                    "attributes": entity.get("attributes") or {},
                },
            })
        return json.dumps(payload, ensure_ascii=False)

    def _make_predicate_batch_message(self, items: List[Dict[str, Any]]) -> str:
        """Build a single user message containing all predicate pairs as a JSON array."""
        payload = []
        for item in items:
            entity      = item["entity"]
            canonical   = item["canonical"]
            old_payload = canonical.get("payload", {})
            payload.append({
                "id": item["id"],
                "existing": {
                    "predicate":    old_payload.get("name_en") or old_payload.get("name", ""),
                    "predicate_raw": old_payload.get("name", ""),
                    "id":           canonical.get("id"),
                    "score":        round(canonical.get("score", 0.0), 3),
                },
                "incoming": {
                    "predicate":    entity.get("name_en") or entity["name"],
                    "predicate_raw": entity["name"],
                },
            })
        return json.dumps(payload, ensure_ascii=False)

    def _parse_batch_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM's JSON array response into a list of decision dicts.

        Handles both a bare array and an object with a top-level array value,
        and strips markdown fences if present.
        """
        try:
            text = response.strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text.strip())
            data = json.loads(text)
            if isinstance(data, list):
                return data
            # Sometimes the model wraps the array: {"decisions": [...]}
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
        except Exception:
            pass
        # Last resort: try parse_json_with_repair
        try:
            data = parse_json_with_repair(response)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
        except Exception:
            pass
        print(f"Warning: Could not parse batch LLM response as array: {response[:200]}")
        return []

    # ------------------------------------------------------------------
    # Within-batch deduplication
    # ------------------------------------------------------------------

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        try:
            import numpy as np
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception:
            dot = sum(a * b for a, b in zip(vec1, vec2))
            m1  = sum(a * a for a in vec1) ** 0.5
            m2  = sum(b * b for b in vec2) ** 0.5
            return dot / (m1 * m2) if m1 and m2 else 0.0

    def _dedup_within_batch(
        self,
        entities: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> Tuple[List[int], List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        """Greedy O(n²) within-batch dedup.

        Returns:
            rep_indices:     Indices of cluster representatives → sent to Qdrant.
            alias_pairs:     (alias_idx, rep_idx, score) — score >= similarity_threshold.
            uncertain_pairs: (item_idx,  rep_idx, score) — LLM_REFINEMENT_THRESHOLD <= score < threshold.
        """
        n = len(entities)
        if n <= 1:
            return list(range(n)), [], []

        rep_indices:     List[int]                   = []
        alias_pairs:     List[Tuple[int, int, float]] = []
        uncertain_pairs: List[Tuple[int, int, float]] = []

        for i in range(n):
            best_rep_idx: Optional[int] = None
            best_score = 0.0
            text_i     = entities[i].get("name_en") or entities[i]["name"]
            has_num_i  = bool(re.search(r"\d", text_i))

            for rep_idx in rep_indices:
                score = self._cosine_similarity(embeddings[i], embeddings[rep_idx])
                if score >= LLM_REFINEMENT_THRESHOLD and score > best_score:
                    best_score   = score
                    best_rep_idx = rep_idx

            if best_rep_idx is None:
                rep_indices.append(i)
            else:
                text_rep    = entities[best_rep_idx].get("name_en") or entities[best_rep_idx]["name"]
                has_num_rep = bool(re.search(r"\d", text_rep))
                auto_threshold = 0.999 if (has_num_i or has_num_rep) else self.similarity_threshold
                if best_score >= auto_threshold:
                    alias_pairs.append((i, best_rep_idx, best_score))
                else:
                    uncertain_pairs.append((i, best_rep_idx, best_score))

        return rep_indices, alias_pairs, uncertain_pairs

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(
        self, collection_name: str, entity: Dict[str, Any], is_predicate: bool
    ) -> tuple:
        if is_predicate:
            return (collection_name, entity["name"], entity.get("community_id"))
        return (collection_name, entity["name"])

    def _make_cache_entry(
        self,
        entity: Dict[str, Any],
        canonical: str,
        canonical_id: str,
        is_new: bool,
        is_canonical: bool,
        is_predicate: bool,
        is_label: bool,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "original":     entity["name"],
            "canonical":    canonical,
            "canonical_id": canonical_id,
            "is_new":       is_new,
            "is_canonical": is_canonical,
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

    # ------------------------------------------------------------------
    # Core collection processing
    # ------------------------------------------------------------------

    def _process_collection(
        self,
        collection_name: str,
        entity_list: List[Dict[str, Any]],
        is_predicate: bool,
        is_label: bool,
        entity_cache: Dict,
        canonical_change_log: List,
    ) -> None:
        """Full pipeline for one registry collection.

        Steps
        -----
        1. Embed all items in one batch call.
        2. Within-batch dedup (cosine similarity, O(n²)).
        3. Batch-query Qdrant for the representatives only.
        4. Classify each representative: new / merge / replace.
        5. Fire all LLM calls concurrently for the uncertain subset.
        6. Collect all decisions, then flush in at most three bulk upserts:
             a. New canonicals (unmatched + LLM-decided "different")
             b. Aliases / merges (same entity, existing stays canonical)
             c. Replacements    (same entity, incoming is better canonical)
        7. Process within-batch aliases (high-confidence).
        8. Process within-batch uncertain pairs via LLM.
        """
        kind = "label" if is_label else ("predicate" if is_predicate else "entity")
        n    = len(entity_list)
        print(f"[{collection_name}] Processing {n} {kind}s")

        # ── Step 1: Embed ─────────────────────────────────────────────────────
        embed_texts   = [
            e["name"] if is_label else (e.get("name_en") or e["name"])
            for e in entity_list
        ]
        en_embeddings = self._get_embeddings_batch(embed_texts)
        print(f"[{collection_name}]   Embedded {n} texts")

        # ── Step 2: Within-batch dedup ────────────────────────────────────────
        rep_indices, alias_pairs, uncertain_local_pairs = self._dedup_within_batch(
            entity_list, en_embeddings
        )
        rep_entities   = [entity_list[i]   for i in rep_indices]
        rep_embeddings = [en_embeddings[i] for i in rep_indices]

        if alias_pairs:
            print(
                f"[{collection_name}]   Within-batch dedup: "
                f"{len(alias_pairs)} high-conf alias(es) → {len(rep_entities)} representative(s)"
            )
            for alias_idx, rep_idx, score in alias_pairs:
                print(f"    '{entity_list[alias_idx]['name']}' → '{entity_list[rep_idx]['name']}' (score={score:.3f})")
        if uncertain_local_pairs:
            print(f"[{collection_name}]   {len(uncertain_local_pairs)} within-batch pair(s) flagged for LLM")

        # ── Step 3: Batch-query Qdrant (representatives only) ─────────────────
        rep_community_ids: Optional[List[Optional[str]]] = (
            [e.get("community_id") for e in rep_entities] if is_predicate else None
        )
        all_matches = self._query_qdrant_canonical_batch(
            collection_name, rep_embeddings, limit=1,
            community_ids=rep_community_ids,
            score_threshold=LLM_REFINEMENT_THRESHOLD,
        )
        matched_count = sum(1 for v in all_matches.values() if v)
        print(f"[{collection_name}]   Qdrant: {matched_count} matches (threshold={LLM_REFINEMENT_THRESHOLD})")

        # ── Step 4: Classify representatives ─────────────────────────────────
        unmatched:      List[Dict[str, Any]]                    = []
        llm_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

        for q_idx, entity in enumerate(rep_entities):
            matches = all_matches.get(q_idx, [])
            if not matches:
                unmatched.append(entity)
            else:
                llm_candidates.append((entity, matches[0]))

        print(
            f"[{collection_name}]   Classified: "
            f"{len(unmatched)} new | {len(llm_candidates)} LLM candidates"
        )

        # ── Step 5: Fire all LLM calls concurrently ───────────────────────────
        entity_type = "predicate" if is_predicate else ("label" if is_label else "entity")
        decisions: Dict[str, Dict[str, Any]] = {}
        if llm_candidates:
            print(f"[{collection_name}]   Running {len(llm_candidates)} LLM call(s) concurrently...")
            decisions = self._run_llm_batch(llm_candidates, is_predicate, entity_type)
            outcomes = {"new": 0, "merged": 0, "replaced": 0}
            for entity, _ in llm_candidates:
                d = decisions.get(entity["name"], {})
                is_same      = d.get("are_same_entity", d.get("are_equivalent", True))
                new_is_canon = d.get("new_is_canonical", False)
                if not is_same:
                    outcomes["new"] += 1
                elif not new_is_canon:
                    outcomes["merged"] += 1
                else:
                    outcomes["replaced"] += 1
            print(
                f"[{collection_name}]   LLM: "
                f"{outcomes['new']} new | {outcomes['merged']} merged | {outcomes['replaced']} replaced"
            )

        # ── Step 6: Bulk upserts ──────────────────────────────────────────────
        #
        # 6a — Collect all new canonicals (unmatched + LLM "different") ───────
        new_canonical_entities: List[Dict[str, Any]] = [
            {**e, "is_canonical": True} for e in unmatched
        ]
        alias_entities:   List[Tuple[Dict, Dict, Dict]] = []  # (entity, canonical_match, decision)
        replace_entities: List[Tuple[Dict, Dict, Dict]] = []

        for entity, canonical_match in llm_candidates:
            decision     = decisions.get(entity["name"], {})
            is_same      = decision.get("are_same_entity", decision.get("are_equivalent", True))
            new_is_canon = decision.get("new_is_canonical", False)
            if not is_same:
                new_canonical_entities.append({**entity, "is_canonical": True})
            elif not new_is_canon:
                alias_entities.append((entity, canonical_match, decision))
            else:
                replace_entities.append((entity, canonical_match, decision))

        # 6b — Single bulk upsert for all new canonicals ───────────────────────
        if new_canonical_entities:
            upserted_ids = self._batch_upsert_entities(new_canonical_entities, collection_name)
            for entity in new_canonical_entities:
                canonical_name = entity.get("name_en") or entity["name"]
                canonical_id   = upserted_ids.get(entity["name"])
                entity_cache[self._cache_key(collection_name, entity, is_predicate)] = (
                    self._make_cache_entry(
                        entity, canonical_name, canonical_id,
                        is_new=True, is_canonical=True,
                        is_predicate=is_predicate, is_label=is_label,
                    )
                )
            print(f"[{collection_name}]   Upserted {len(new_canonical_entities)} new canonical(s)")

        # 6c — Single bulk upsert for all aliases / merges ────────────────────
        if alias_entities:
            alias_points = [
                {**entity, "is_canonical": False, "canonical_id": canonical_match["id"]}
                for entity, canonical_match, _ in alias_entities
            ]
            self._batch_upsert_entities(alias_points, collection_name)
            for entity, canonical_match, _ in alias_entities:
                old_payload    = canonical_match.get("payload", {})
                existing_name  = old_payload.get("name_en") or old_payload.get("name", "")
                canonical_id   = old_payload.get("canonical_id", canonical_match["id"])
                entity_cache[self._cache_key(collection_name, entity, is_predicate)] = (
                    self._make_cache_entry(
                        entity, existing_name, canonical_id,
                        is_new=False, is_canonical=False,
                        is_predicate=is_predicate, is_label=is_label,
                    )
                )
            print(f"[{collection_name}]   Upserted {len(alias_entities)} alias(es)")

        # 6d — Replacements: new canonical + demote old ───────────────────────
        if replace_entities:
            replace_points = [{**entity, "is_canonical": True} for entity, _, _ in replace_entities]
            upserted_ids   = self._batch_upsert_entities(replace_points, collection_name)
            for entity, canonical_match, decision in replace_entities:
                new_id         = upserted_ids.get(entity["name"])
                old_id         = canonical_match["id"]
                old_payload    = canonical_match.get("payload", {})
                incoming_name  = entity.get("name_en") or entity["name"]
                canonical_form = decision.get("canonical_form") or incoming_name
                try:
                    self.qdrant_client.set_payload(
                        collection_name=self._qdrant_name(collection_name),
                        payload={"is_canonical": False, "canonical_id": new_id},
                        points=[old_id],
                    )
                except Exception as e:
                    print(f"  Warning: Failed to demote old canonical '{old_payload.get('name')}': {e}")

                canonical_change_log.append({
                    "collection":         collection_name,
                    "old_canonical_id":   old_id,
                    "old_canonical_name": old_payload.get("name"),
                    "new_canonical_id":   new_id,
                    "new_canonical_name": entity["name"],
                    "reasoning":          decision.get("reasoning", ""),
                })
                entity_cache[self._cache_key(collection_name, entity, is_predicate)] = (
                    self._make_cache_entry(
                        entity, canonical_form, new_id,
                        is_new=True, is_canonical=True,
                        is_predicate=is_predicate, is_label=is_label,
                    )
                )
            print(f"[{collection_name}]   Replaced {len(replace_entities)} canonical(s)")

        # ── Step 7: High-confidence within-batch aliases ──────────────────────
        if alias_pairs:
            alias_points_wb: List[Dict[str, Any]] = []
            for alias_idx, rep_idx, _score in alias_pairs:
                alias_entity = entity_list[alias_idx]
                rep_entity   = entity_list[rep_idx]
                rep_cache    = entity_cache.get(
                    self._cache_key(collection_name, rep_entity, is_predicate)
                )
                if rep_cache:
                    canonical_id   = rep_cache["canonical_id"]
                    canonical_name = rep_cache["canonical"]
                    alias_points_wb.append(
                        {**alias_entity, "is_canonical": False, "canonical_id": canonical_id}
                    )
                    entity_cache[self._cache_key(collection_name, alias_entity, is_predicate)] = (
                        self._make_cache_entry(
                            alias_entity, canonical_name, canonical_id,
                            is_new=False, is_canonical=False,
                            is_predicate=is_predicate, is_label=is_label,
                        )
                    )
                    print(f"    '{alias_entity['name']}' → alias of '{canonical_name}' (high-conf)")
                else:
                    print(f"  Warning: rep '{rep_entity['name']}' not in cache; treating '{alias_entity['name']}' as new canonical")
                    self._upsert_single_new_canonical(
                        alias_entity, collection_name, is_predicate, is_label, entity_cache
                    )

            if alias_points_wb:
                self._batch_upsert_entities(alias_points_wb, collection_name)
            print(f"[{collection_name}]   Upserted {len(alias_pairs)} within-batch alias(es)")

        # ── Step 8: LLM for uncertain within-batch pairs ──────────────────────
        if uncertain_local_pairs:
            print(f"[{collection_name}]   Running LLM for {len(uncertain_local_pairs)} uncertain within-batch pair(s)...")

            wb_pairs: List[Tuple[Dict, Dict]] = []
            wb_synthetics: List[Dict]          = []  # synthetic "canonical" dict per pair
            for item_idx, rep_idx, score in uncertain_local_pairs:
                item_entity = entity_list[item_idx]
                rep_entity  = entity_list[rep_idx]
                rep_cache   = entity_cache.get(
                    self._cache_key(collection_name, rep_entity, is_predicate)
                )
                if not rep_cache:
                    print(f"  Warning: rep '{rep_entity['name']}' not in cache; treating '{item_entity['name']}' as new canonical")
                    self._upsert_single_new_canonical(
                        item_entity, collection_name, is_predicate, is_label, entity_cache
                    )
                    continue

                synthetic = {
                    "payload": {
                        "name":       rep_entity["name"],
                        "name_en":    rep_entity.get("name_en"),
                        "attributes": rep_entity.get("attributes", {}),
                    },
                    "id":    rep_cache.get("canonical_id"),
                    "score": score,
                }
                wb_pairs.append((item_entity, synthetic))
                wb_synthetics.append({"item": item_entity, "rep_cache": rep_cache, "score": score})

            # Fire LLM calls concurrently
            wb_decisions = self._run_llm_batch(wb_pairs, is_predicate, entity_type)

            wb_new_canonicals:  List[Dict] = []
            wb_alias_points:    List[Dict] = []
            for (item_entity, synthetic), meta in zip(wb_pairs, wb_synthetics):
                decision  = wb_decisions.get(item_entity["name"], {})
                is_same   = decision.get("are_same_entity", decision.get("are_equivalent", False))
                rep_cache = meta["rep_cache"]

                if is_same:
                    canonical_id   = rep_cache["canonical_id"]
                    canonical_name = rep_cache["canonical"]
                    wb_alias_points.append(
                        {**item_entity, "is_canonical": False, "canonical_id": canonical_id}
                    )
                    entity_cache[self._cache_key(collection_name, item_entity, is_predicate)] = (
                        self._make_cache_entry(
                            item_entity, canonical_name, canonical_id,
                            is_new=False, is_canonical=False,
                            is_predicate=is_predicate, is_label=is_label,
                        )
                    )
                    print(f"    LLM: '{item_entity['name']}' → alias of '{canonical_name}'")
                else:
                    wb_new_canonicals.append({**item_entity, "is_canonical": True})
                    print(f"    LLM: '{item_entity['name']}' → new canonical")

            # Bulk flush within-batch LLM decisions
            if wb_new_canonicals:
                ids = self._batch_upsert_entities(wb_new_canonicals, collection_name)
                for entity in wb_new_canonicals:
                    canonical_name = entity.get("name_en") or entity["name"]
                    entity_cache[self._cache_key(collection_name, entity, is_predicate)] = (
                        self._make_cache_entry(
                            entity, canonical_name, ids.get(entity["name"]),
                            is_new=True, is_canonical=True,
                            is_predicate=is_predicate, is_label=is_label,
                        )
                    )

            if wb_alias_points:
                self._batch_upsert_entities(wb_alias_points, collection_name)

    def _upsert_single_new_canonical(
        self,
        entity: Dict[str, Any],
        collection_name: str,
        is_predicate: bool,
        is_label: bool,
        entity_cache: Dict,
    ) -> None:
        """Insert a single entity as a new canonical and populate the cache."""
        ids            = self._batch_upsert_entities([{**entity, "is_canonical": True}], collection_name)
        canonical_name = entity.get("name_en") or entity["name"]
        canonical_id   = ids.get(entity["name"])
        entity_cache[self._cache_key(collection_name, entity, is_predicate)] = (
            self._make_cache_entry(
                entity, canonical_name, canonical_id,
                is_new=True, is_canonical=True,
                is_predicate=is_predicate, is_label=is_label,
            )
        )

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _dedup_evidence_within_batch(
        self,
        records: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """Greedy within-batch dedup for evidence records."""
        n = len(records)
        if n <= 1:
            return list(range(n)), []

        rep_indices: List[int]                   = []
        dup_pairs:   List[Tuple[int, int, float]] = []

        for i in range(n):
            best_rep_idx: Optional[int] = None
            best_score = 0.0
            text_i     = records[i].get("evidence_quote_en") or records[i].get("evidence_quote", "")
            has_num_i  = bool(re.search(r"\d", text_i))

            for rep_idx in rep_indices:
                text_rep    = records[rep_idx].get("evidence_quote_en") or records[rep_idx].get("evidence_quote", "")
                has_num_rep = bool(re.search(r"\d", text_rep))
                threshold   = 0.999 if (has_num_i or has_num_rep) else self.similarity_threshold
                score       = self._cosine_similarity(embeddings[i], embeddings[rep_idx])
                if score >= threshold and score > best_score:
                    best_score   = score
                    best_rep_idx = rep_idx

            if best_rep_idx is None:
                rep_indices.append(i)
            else:
                dup_pairs.append((i, best_rep_idx, best_score))

        return rep_indices, dup_pairs

    def _upsert_evidence_triples(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed and upsert evidence into the evidence collection.

        Triples are grouped by evidence_quote.  Each unique quote becomes one
        Qdrant point; existing points have their entity/predicate lists merged.
        """
        from qdrant_client.http import models as qm

        evidence_spec       = self._role_to_spec_key.get("evidence_quote", "evidence_registry")
        vec_en, vec_th      = self._vector_names(evidence_spec)
        sparse_vec_en, sparse_vec_th = self._sparse_vector_names(evidence_spec)
        evidence_collection = self._qdrant_name(evidence_spec)

        # ── Group triples by evidence_quote ──────────────────────────────────
        groups: Dict[str, Dict[str, Any]] = {}
        total_triples = 0
        for chunk in chunks:
            for triple in chunk.get("triples", []):
                total_triples += 1
                props  = triple.get("properties", {})
                eq     = props.get("evidence_quote", "")
                eq_en  = props.get("evidence_quote_en", "")
                if eq not in groups:
                    groups[eq] = {
                        "evidence_quote":    eq,
                        "evidence_quote_en": eq_en,
                        "entities":          set(),
                        "predicates":        set(),
                        "community_id":      props.get("community_id"),
                    }
                groups[eq]["entities"].add(triple["subject"]["name"])
                groups[eq]["entities"].add(triple["object"]["name"])
                groups[eq]["predicates"].add(triple["predicate"])
                if eq_en and not groups[eq]["evidence_quote_en"]:
                    groups[eq]["evidence_quote_en"] = eq_en

        if not groups:
            print("No evidence triples to upsert.")
            return

        records = list(groups.values())
        print(f"Grouped {total_triples} triples into {len(records)} unique evidence records")

        # ── Embed (dense) ─────────────────────────────────────────────────────
        en_texts = [r["evidence_quote_en"] if r["evidence_quote_en"].strip() else "[no evidence]" for r in records]
        th_texts = [r["evidence_quote"]    if r["evidence_quote"].strip()    else "[no evidence]" for r in records]
        print(f"Embedding {len(records)} evidence records (dual-vector)...")
        en_embeddings = self._get_evidence_embeddings_batch(en_texts)
        th_embeddings = self._get_evidence_embeddings_batch(th_texts)

        # ── Compute sparse vectors ────────────────────────────────────────────
        # FIX 4: Use only the natural-language evidence quote as the sparse
        # document text.  Entity/predicate labels belong in the dense (semantic)
        # space; the sparse (keyword) index must share token vocabulary with
        # the natural-language queries issued at retrieval time.
        en_sparse: List[Dict[str, Any]] = []
        th_sparse: List[Dict[str, Any]] = []
        if sparse_vec_en:
            print(f"Computing sparse vectors ({sparse_vec_en}, {sparse_vec_th})...")
            en_doc_texts = [r["evidence_quote_en"] or "[no evidence]" for r in records]
            th_doc_texts = [r["evidence_quote"]    or "[no evidence]" for r in records]

            en_vocab  = self._build_vocab(en_doc_texts)
            th_vocab  = self._build_vocab(th_doc_texts)
            en_sparse = self._compute_sparse_tf_batch(en_doc_texts, en_vocab)
            th_sparse = self._compute_sparse_tf_batch(th_doc_texts, th_vocab)

        # ── Batch-query Qdrant for exact matches (score 1.0) ──────────────────
        merged_count  = 0
        new_indices: List[int] = []
        print(f"  Querying {len(records)} exact matches (batched)...")

        requests_batch = [
            qm.QueryRequest(
                query=qm.NearestQuery(nearest=en_embeddings[i]),
                using=vec_en,
                limit=1,
                with_payload=True,
                score_threshold=1.0,
            )
            for i in range(len(records))
        ]
        batch_results: List[Any] = []
        for chunk_start in range(0, len(requests_batch), QUERY_BATCH_SIZE):
            chunk = requests_batch[chunk_start:chunk_start + QUERY_BATCH_SIZE]
            try:
                batch_results.extend(
                    self.qdrant_client.query_batch_points(
                        collection_name=evidence_collection,
                        requests=chunk,
                    )
                )
            except Exception as e:
                print(f"  Warning: Batch query failed: {e} — treating as no matches")
                batch_results.extend([None] * len(chunk))

        for i, result in enumerate(batch_results):
            if result and result.points:
                existing         = result.points[0]
                existing_payload = existing.payload or {}
                merged_entities  = set(existing_payload.get("entities",   [])) | records[i]["entities"]
                merged_preds     = set(existing_payload.get("predicates", [])) | records[i]["predicates"]
                self.qdrant_client.set_payload(
                    collection_name=evidence_collection,
                    payload={
                        "entities":   sorted(merged_entities),
                        "predicates": sorted(merged_preds),
                    },
                    points=[existing.id],
                )
                merged_count += 1
            else:
                new_indices.append(i)

        if merged_count:
            print(f"  Merged {merged_count} records into existing points")

        if not new_indices:
            print("All evidence records merged into existing points.")
            return

        # ── Build and upsert new points ───────────────────────────────────────
        points: List[qm.PointStruct] = []
        for i in new_indices:
            record   = records[i]
            point_id = self._generate_uuid(record["evidence_quote"] or f"evidence_{i}")
            payload: Dict[str, Any] = {
                "entities":          sorted(record["entities"]),
                "predicates":        sorted(record["predicates"]),
                "evidence_quote":    record["evidence_quote"],
                "evidence_quote_en": record["evidence_quote_en"],
            }
            if record["community_id"]:
                payload["community_id"] = record["community_id"]

            vectors: Dict[str, Any] = {vec_en: en_embeddings[i], vec_th: th_embeddings[i]}
            if sparse_vec_en and en_sparse:
                sp_en = en_sparse[i]
                if sp_en["indices"]:
                    vectors[sparse_vec_en] = qm.SparseVector(indices=sp_en["indices"], values=sp_en["values"])
            if sparse_vec_th and th_sparse:
                sp_th = th_sparse[i]
                if sp_th["indices"]:
                    vectors[sparse_vec_th] = qm.SparseVector(indices=sp_th["indices"], values=sp_th["values"])

            points.append(qm.PointStruct(id=point_id, vector=vectors, payload=payload))

        print(f"Upserting {len(points)} new evidence points...")
        self._qdrant_upsert_with_retry(
            evidence_collection, points, chunk_size=EVIDENCE_UPSERT_CHUNK_SIZE
        )
        print(f"✅ Evidence upsert complete ({len(points)} new, {merged_count} merged).")

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------

    def _build_refined_output(
        self,
        triples_data: Dict[str, Any],
        entity_cache: Dict[tuple, Dict[str, Any]],
        canonical_change_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble the refined triples output dict using English canonical forms."""
        entity_spec    = self._role_to_spec_key.get("entity",    "entity_registry")
        predicate_spec = self._role_to_spec_key.get("predicate", "predicate_registry")
        label_spec     = self._role_to_spec_key.get("label",     "label_registry")

        refined_chunks = []
        for chunk_data in triples_data.get("chunks", []):
            chunk_id        = chunk_data["chunk_id"]
            refined_triples = []

            for triple in chunk_data.get("triples", []):
                subj_raw = triple["subject"]
                obj_raw  = triple["object"]

                community_id  = triple.get("properties", {}).get("community_id")
                predicate_key = (predicate_spec, triple["predicate"], community_id)

                refined_subject = entity_cache.get(
                    (entity_spec, subj_raw["name"]),
                    {
                        "original":     subj_raw["name"],
                        "canonical":    subj_raw.get("name_en") or subj_raw["name"],
                        "canonical_id": None,
                        "is_new":       True,
                        "label":        subj_raw.get("label", ""),
                        "name_en":      subj_raw.get("name_en"),
                        "attributes":   subj_raw.get("attributes", {}),
                        "attributes_en": subj_raw.get("attributes_en", {}),
                    },
                )
                refined_predicate = entity_cache.get(
                    predicate_key,
                    {
                        "original":  triple["predicate"],
                        "canonical": triple.get("predicate_en") or triple["predicate"],
                        "canonical_id": None,
                        "is_new":    True,
                        "label":     "predicate",
                        "name_en":   triple.get("predicate_en"),
                    },
                )
                refined_object = entity_cache.get(
                    (entity_spec, obj_raw["name"]),
                    {
                        "original":     obj_raw["name"],
                        "canonical":    obj_raw.get("name_en") or obj_raw["name"],
                        "canonical_id": None,
                        "is_new":       True,
                        "label":        obj_raw.get("label", ""),
                        "name_en":      obj_raw.get("name_en"),
                        "attributes":   obj_raw.get("attributes", {}),
                        "attributes_en": obj_raw.get("attributes_en", {}),
                    },
                )
                refined_obj_label = entity_cache.get(
                    (label_spec, obj_raw.get("label", "")),
                    {
                        "original":  obj_raw.get("label", ""),
                        "canonical": obj_raw.get("label", ""),
                        "canonical_id": None,
                        "is_new":    True,
                        "label":     "label",
                    },
                )

                refined_triple: Dict[str, Any] = {
                    "subject": {
                        "name":          refined_subject["canonical"],
                        "original_name": subj_raw["name"],
                        "label":         refined_subject.get("label", ""),
                        "canonical_id":  refined_subject.get("canonical_id"),
                    },
                    "predicate":          refined_predicate["canonical"],
                    "original_predicate": triple["predicate"],
                    "object": {
                        "name":          refined_object["canonical"],
                        "original_name": obj_raw["name"],
                        "label":         refined_obj_label["canonical"],
                        "canonical_id":  refined_object.get("canonical_id"),
                    },
                    "properties": triple.get("properties", {}),
                    "chunk_id":   chunk_id,
                }

                # Bilingual metadata
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

            refined_chunks.append({"chunk_id": chunk_id, "triples": refined_triples})

        return {
            "source_file":        triples_data.get("source_file"),
            "total_chunks":       len(refined_chunks),
            "total_triples":      sum(len(c["triples"]) for c in refined_chunks),
            "llm_provider":       self.llm_provider,
            "llm_model":          self.llm_model,
            "refinement_applied": True,
            "canonical_change_log": canonical_change_log,
            "chunks":             refined_chunks,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def refine_triples(self, triples_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine all triples using Qdrant for entity resolution.

        Resolution is always performed in English embedding space.
        Thai forms are stored only in Qdrant payloads for downstream lookup.
        The KG output uses English canonical forms throughout.
        """
        self._ensure_collections_exist()

        entity_spec    = self._role_to_spec_key.get("entity",    "entity_registry")
        predicate_spec = self._role_to_spec_key.get("predicate", "predicate_registry")
        label_spec     = self._role_to_spec_key.get("label",     "label_registry")

        # ── Collect unique items per registry ─────────────────────────────────
        registry_items: Dict[str, Dict] = {
            entity_spec: {}, predicate_spec: {}, label_spec: {}
        }
        for chunk_data in triples_data.get("chunks", []):
            for triple in chunk_data.get("triples", []):
                subj, obj = triple["subject"], triple["object"]

                for node in (subj, obj):
                    name = node["name"]
                    if name not in registry_items[entity_spec]:
                        registry_items[entity_spec][name] = {
                            "name":          name,
                            "name_en":       node.get("name_en"),
                            "label":         node.get("label", ""),
                            "attributes":    node.get("attributes", {}),
                            "attributes_en": node.get("attributes_en", {}),
                        }

                pred    = triple["predicate"]
                pred_en = triple.get("predicate_en")
                cid     = triple.get("properties", {}).get("community_id")
                pred_key = (pred, cid)
                if pred_key not in registry_items[predicate_spec]:
                    registry_items[predicate_spec][pred_key] = {
                        "name":         pred,
                        "name_en":      pred_en,
                        "community_id": cid,
                    }

                for lbl in (subj.get("label", ""), obj.get("label", "")):
                    if lbl and lbl not in registry_items[label_spec]:
                        registry_items[label_spec][lbl] = {"name": lbl}

        total_items = sum(len(v) for v in registry_items.values())
        print(f"Collected {total_items} unique items across {len(registry_items)} registries")

        # ── Evidence registry first ────────────────────────────────────────────
        print("📎 Processing evidence registry...")
        self._upsert_evidence_triples(triples_data.get("chunks", []))
        print("✅ Evidence registry processed\n")

        # ── Entity / predicate / label registries ─────────────────────────────
        entity_cache:         Dict[tuple, Dict[str, Any]] = {}
        canonical_change_log: List[Dict[str, Any]]        = []

        for collection_name, items in registry_items.items():
            if not items:
                continue
            is_pred = collection_name == predicate_spec
            is_lbl  = collection_name == label_spec
            self._process_collection(
                collection_name,
                list(items.values()),
                is_predicate=is_pred,
                is_label=is_lbl,
                entity_cache=entity_cache,
                canonical_change_log=canonical_change_log,
            )
            # No artificial sleep between collections.

        if canonical_change_log:
            print(f"Canonical replacements this run: {len(canonical_change_log)}")

        return self._build_refined_output(triples_data, entity_cache, canonical_change_log)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save_refined_triples(
        self,
        refined_data: Dict[str, Any],
        output_path: str,
    ) -> str:
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
    """Refine triples from a JSON file and save the result."""
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
        p           = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_refined{p.suffix}")

    return refiner.save_refined_triples(refined_data, output_path)