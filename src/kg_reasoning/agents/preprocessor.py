"""PreProcessor agent for multi-agent reasoning system.

Runs a deterministic pipeline before the orchestrator, in this exact order:
1. Query expansion (fast LLM)
2. Evidence search + community metadata fetch — IN PARALLEL (two independent
   Qdrant calls, neither depends on the other's result)
3. LLM resolves which community_id(s) from the metadata are relevant to the query
4. HARD FILTER: drop every evidence point whose community_id is not in the
   resolved set, BEFORE any further LLM call sees it
5. Source structuring (indexed entity/predicate lists per surviving evidence point,
   grouped by community_id)
6. Stage A — LLM filters evidence points using BOTH the quote text AND the
   community_id together (not quote alone)
7. Stage B — LLM filters entities/predicates from the surviving evidence,
   again scoped per-community-group, against query + community_id
8. Canonical ID lookup (raw entity/predicate names → canonical_id, preserving
   the raw→canonical linkage) — runs in parallel with step 9
9. S3 chunk fetch: for every surviving evidence point's (community_id, chunk_id),
   fetch the actual source chunk text from S3 — NOT Qdrant — at
   cmu_ci_documents/{community_id}/chunk_{chunk_id:03d}.txt
10. Assemble output with query type classification

WHY COMMUNITY RESOLUTION HAPPENS BEFORE ANY EVIDENCE FILTERING:
Community metadata and evidence are two unrelated Qdrant collections, fetched
concurrently so neither blocks the other. Once both are back, community_id
becomes a HARD gate: evidence from communities the LLM didn't resolve as
relevant is dropped entirely before Stage A ever sees it. This is different
from "tag and let the LLM weigh it" — a community-scoped question (e.g.
"What happens in หมู่ 2?") should never let a topically-similar passage from
หมู่ 5 leak through just because its text looked relevant in isolation.

WHY TWO LLM CALLS INSTEAD OF ONE (Stage A / Stage B):
Stage A (evidence selection) decides relevance per PASSAGE using quote text +
community_id. Stage B (entity/predicate filtering) only runs over passages
Stage A already kept, and decides relevance per ENTITY/PREDICATE INDEX using
explicit criteria (named in question / matches what the question asks about /
bridges two relevant entities) plus the same community_id signal, grouped by
community so the model never has to cross-reason between unrelated villages
in a single judgment.

WHY S3, NOT QDRANT, FOR CHUNK TEXT:
Qdrant's evidence_registry only stores a short evidence_quote excerpt — the
sentence the entity/predicate was extracted from, not the full source chunk.
The actual chunk file lives in S3, keyed by the SAME community_id and
chunk_id that the surviving evidence points already carry. This step reads
it directly via chunk_tools.fetch_s3_chunks — no LLM, no extra Qdrant call.
"""

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from kg_extractor.utils.model_setup import get_reasoning_llm, PREPROCESSING_MODEL
from kg_reasoning.agents.tools.qdrant_tools import _get_manager
from kg_reasoning.agents.tools.chunk_tools import fetch_s3_chunks
from kg_reasoning.prompts.preprocessor_prompts import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    SOURCE_SELECTION_SYSTEM_PROMPT,
    ENTITY_PREDICATE_FILTER_SYSTEM_PROMPT,
    COMMUNITY_FILTER_SYSTEM_PROMPT
)


# ---------------------------------------------------------------------------
# Community-keyword heuristic (no LLM needed)
# ---------------------------------------------------------------------------
_COMMUNITY_KEYWORDS = re.compile(
    r"moo\s*\d|หมู่\s*\d|village|community|หมู่บ้าน|ชุมชน",
    re.IGNORECASE,
)



class PreProcessor:
    """Deterministic pre-processor that resolves canonical IDs before the
    orchestrator runs, replacing multiple ReAct tool-call round-trips with
    two fast LLM calls plus embedding lookups."""

    def __init__(
        self,
        llm_provider: str = "",
        llm_model: str = PREPROCESSING_MODEL,
    ):
        # Low-temperature LLM for the deterministic filtering/classification
        # calls (community resolution, Stage A source selection, Stage B
        # entity/predicate filtering).
        self.llm = get_reasoning_llm(model=llm_model, temperature=0.1)
        # Separate, higher-temperature LLM used ONLY for query expansion —
        # diversity of rewrites matters there, unlike the filtering calls.
        # get_reasoning_llm caches by (provider, model, temperature,
        # max_tokens), so this stays a distinct instance from self.llm.
        self.llm_expansion = get_reasoning_llm(model=llm_model, temperature=0.7)
        self.qdrant_manager = _get_manager()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, user_query: str) -> Dict[str, Any]:
        """Execute the full pipeline.

        Returns a dict with keys:
            expanded_query, entity_ids, predicate_ids, community_ids,
            entity_details, predicate_details, raw_entity_names,
            raw_predicate_names, chunk_ids, chunk_details, chunk_texts,
            query_type, needs_pathing, needs_community
        """
        # Step 1 – expand
        expanded = self._expand_query(user_query)
        print(f"  📝 Expanded query: {expanded[:120]}...")

        # Step 2 – evidence search AND community metadata fetch, IN PARALLEL.
        # Neither depends on the other's output — evidence search needs only
        # the expanded query embedding, metadata fetch is a plain scroll
        # over metadata_registry. Running them concurrently means community
        # resolution is ready the moment evidence search finishes, instead
        # of adding its own latency afterward.
        with ThreadPoolExecutor(max_workers=2) as pool:
            evidence_future = pool.submit(self._search_evidence, expanded)
            metadata_future = pool.submit(self.qdrant_manager.get_all_community_metadata)
            evidence = evidence_future.result()
            all_metadata = metadata_future.result()
        print(f"  🔎 Evidence points: {len(evidence)}")
        if not evidence:
            return self._empty_result(expanded)

        # Build a community_id -> human-readable label map from the SAME
        # metadata fetch used for resolution — no extra Qdrant call. This is
        # what lets the aggregator/synthesizer refer to "Village 2" instead
        # of the opaque internal community_id string, and is the only
        # reliable way to connect a user's "village 1"/"village 2" phrasing
        # to the right community_id, since internal IDs like หมู่-1_village-1
        # vs หมู่-2_village2 have inconsistent formatting (note the missing
        # hyphen before "2") that an LLM can plausibly misread.
        community_labels = self._build_community_labels(all_metadata)

        # Step 3 – LLM resolves which community_id(s) are relevant
        resolved_community_ids = self._resolve_community_ids(
            user_query, all_metadata
        )
        print(f"  🏘️  Resolved community IDs: {resolved_community_ids}")

        # Step 4 – HARD FILTER: drop evidence from non-relevant communities
        # before any further LLM call sees it. resolved_community_ids is
        # empty only when resolution itself failed (see _resolve_community_ids'
        # docstring) — in that case skip the filter rather than dropping
        # everything, since an empty set there means "couldn't determine",
        # not "nothing is relevant".
        if resolved_community_ids:
            before = len(evidence)
            allowed = set(resolved_community_ids)
            evidence = [e for e in evidence if e.get("community_id") in allowed]
            print(f"  ✂️  Community filter: {before} → {len(evidence)} evidence points")
            if not evidence:
                print("  ⚠️  No evidence left after community filtering")
                return self._empty_result(expanded)
        else:
            print("  ⚠️  Community resolution returned nothing — skipping hard filter")

        # Step 5 – structure sources, grouped by community_id
        sources, community_ids = self._structure_sources(evidence)
        print(f"  📋 Structured {len(sources)} sources, {len(community_ids)} communities")

        # Step 6 – Stage A: evidence-point selection using quote + community_id together
        try:
            selected_keys = self._select_sources(user_query, sources)
            print(f"  📌 Selected {len(selected_keys)}/{len(sources)} sources as relevant")
        except Exception as e:
            # Inclusive fallback: treat every source as selected so Stage B
            # still runs over everything rather than losing all evidence.
            print(f"  ⚠️  Source selection failed ({e}), keeping all sources")
            selected_keys = list(sources.keys())

        if not selected_keys:
            print("  ⚠️  No sources selected as relevant")
            return self._empty_result(expanded)

        selected_sources = {k: sources[k] for k in selected_keys}

        # Step 7 – Stage B: entity/predicate filtering, grouped per community,
        # scoped to selected sources only
        try:
            filter_result = self._filter_entities_predicates(user_query, selected_sources)
            (filtered_entities, filtered_predicates, filtered_communities, chunk_ids,
             chunk_details, raw_entities_by_community, raw_predicates_by_community) = (
                self._restructure_by_indices(selected_sources, filter_result, community_ids)
            )
            print(f"  🎯 Filtered: {len(filtered_entities)} entities, {len(filtered_predicates)} predicates, {len(chunk_ids)} chunks")
        except Exception as e:
            # Inclusive fallback: use everything from the SELECTED sources
            # (not all sources — Stage A's selection still holds even if
            # Stage B's LLM call fails).
            print(f"  ⚠️  Entity/predicate filter failed ({e}), using all items from selected sources")
            (filtered_entities, filtered_predicates, filtered_communities, chunk_ids,
             chunk_details, raw_entities_by_community, raw_predicates_by_community) = (
                self._all_from_sources(selected_sources, community_ids)
            )

        if not filtered_entities and not filtered_predicates and not chunk_ids:
            print("  ⚠️  No entities, predicates, or chunks after filtering")
            return self._empty_result(expanded)

        # Cap chunk_ids per community BEFORE the S3 fetch — avoids wasted
        # S3 calls for chunks that would get cut later anyway, and keeps
        # downstream context size bounded regardless of how many
        # communities a query spans (see _cap_chunks_per_community).
        chunk_ids = self._cap_chunks_per_community(chunk_ids, chunk_details, user_query)

        # Step 8-9 – canonical ID lookup AND S3 chunk text fetch, IN PARALLEL.
        # Canonicalization (Qdrant) and chunk text retrieval (S3) touch
        # different backends and neither depends on the other's result, so
        # there's no reason to serialize them.
        with ThreadPoolExecutor(max_workers=3) as pool:
            entity_future = pool.submit(self._lookup_entities, filtered_entities)
            predicate_future = pool.submit(self._lookup_predicates, filtered_predicates)
            s3_future = pool.submit(self._fetch_s3_chunk_texts, chunk_ids, chunk_details)
            entity_ids, entity_details = entity_future.result()
            predicate_ids, predicate_details = predicate_future.result()
            chunk_texts = s3_future.result()
        print(f"  🔗 Resolved: {len(entity_ids)} entity IDs, {len(predicate_ids)} predicate IDs")
        print(f"  📄 Fetched {sum(1 for v in chunk_texts.values() if v.get('text'))}/{len(chunk_ids)} S3 chunk texts")

        # Step 9.5 – per-community canonical resolution, for comparison
        # queries (2+ communities). Reuses the SAME entity_details/
        # predicate_details produced above — these already carry raw_names
        # per canonical_id, so building the per-community grouping is a
        # local lookup, not a second batch of Qdrant calls.
        entities_by_community = self._group_canonical_ids_by_community(
            raw_entities_by_community, entity_details
        )
        predicates_by_community = self._group_canonical_ids_by_community(
            raw_predicates_by_community, predicate_details
        )
        if len(filtered_communities) > 1:
            for cid in filtered_communities:
                print(f"  🏘️  {cid}: {len(entities_by_community.get(cid, []))} entities, "
                      f"{len(predicates_by_community.get(cid, []))} predicates")

        # Step 10 – assemble output
        return self._build_output(
            user_query=user_query,
            expanded_query=expanded,
            entity_ids=entity_ids,
            entity_details=entity_details,
            predicate_ids=predicate_ids,
            predicate_details=predicate_details,
            community_ids=filtered_communities,
            community_labels=community_labels,
            chunk_ids=chunk_ids,
            chunk_details=chunk_details,
            chunk_texts=chunk_texts,
            entities_by_community=entities_by_community,
            predicates_by_community=predicates_by_community,
            # Pass the FULL raw filtered lists, not just the subset that
            # resolved to a canonical_id above threshold. A raw name can
            # fail canonical resolution (e.g. a rare surface form scoring
            # below 0.7) yet still be exactly what's stored verbatim in
            # evidence_registry — search_evidence/fetch_chunks need to be
            # able to filter on it regardless of whether canonicalization
            # succeeded.
            all_filtered_raw_entities=filtered_entities,
            all_filtered_raw_predicates=filtered_predicates,
        )

    # ------------------------------------------------------------------
    # Step 1 – Query Expansion
    # ------------------------------------------------------------------

    def _expand_query(self, user_query: str) -> str:
        # Use the higher-temperature expansion LLM — diversity of rewrites
        # matters here, unlike the deterministic filtering calls.
        messages = [
            SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
        response = self.llm_expansion.invoke(messages)
        return response.content.strip()

    # ------------------------------------------------------------------
    # Steps 2-3 – Evidence Search
    # ------------------------------------------------------------------

    def _search_evidence(self, expanded_query: str) -> List[Dict[str, Any]]:
        return self.qdrant_manager.search_evidence(
            query_texts=[expanded_query],
            limit=10,
            threshold=0.2,
        )

    # ------------------------------------------------------------------
    # Step 3 – Community ID Resolution
    # ------------------------------------------------------------------

    def _build_community_labels(
        self, all_metadata: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Build community_id (unique_id) -> human-readable label from
        metadata_registry records, so the synthesizer never has to infer
        "Village 1" from an opaque internal ID string like หมู่-1_village-1.

        Tries several plausible field names defensively since the exact
        metadata_registry schema beyond unique_id isn't something this
        codebase pins down in one place — location_village/location_moo
        were named as examples in COMMUNITY_FILTER_SYSTEM_PROMPT's own
        docstring ("descriptive fields such as ... etc."), not a guaranteed
        complete or exact field list. Falls back to the unique_id itself
        if no descriptive field is found, so a label always exists even
        when metadata is sparse — the caller never has to special-case a
        missing label.

        Args:
            all_metadata: Same metadata_registry payload list used by
                _resolve_community_ids — no extra Qdrant call needed.

        Returns:
            community_id (unique_id) -> label string
        """
        labels: Dict[str, str] = {}
        label_field_candidates = [
            "location_village", "location_moo", "village_name", "village",
            "moo", "document_title", "name", "title",
        ]

        for payload in all_metadata or []:
            uid = payload.get("unique_id")
            if not uid:
                continue

            parts = []
            for field in label_field_candidates:
                value = payload.get(field)
                if value and str(value) not in parts:
                    parts.append(str(value))
                if len(parts) >= 2:
                    # Two descriptive fields (e.g. village + moo number) is
                    # usually enough to be unambiguous without making the
                    # label unwieldy; stop once we have that much.
                    break

            labels[uid] = " / ".join(parts) if parts else uid

        return labels

    def _resolve_community_ids(
        self, user_query: str, all_metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Let the LLM decide which community_ids (unique_ids) are relevant.

        Strategy
        --------
        1. all_metadata is already fetched by run() in parallel with evidence
           search — this method does no I/O of its own besides the LLM call.
        2. Send the full metadata list + user query to the LLM.
        3. LLM returns {"relevant_unique_ids": [...]} — see
           COMMUNITY_FILTER_SYSTEM_PROMPT for the full decision rules:
           - If the query names a specific village/moo, return only matches.
           - If the query has NO location scope, return ALL unique_ids so
             downstream filtering is not restricted to one community.
        4. Extract unique_id values from the LLM response.

        Returns an empty list on any failure. The caller in run() treats an
        empty list as "resolution failed, skip the hard filter" rather than
        "nothing is relevant" — those are different situations and must not
        be conflated, since the latter would wipe out all evidence.
        """
        if not all_metadata:
            return []

        try:
            # Build a compact summary for the LLM: only include fields that
            # help identify the community; drop empty strings to keep the
            # prompt tight.
            summaries = []
            for payload in all_metadata:
                summary = {
                    k: v for k, v in payload.items()
                    if v and k != "vector"
                }
                summaries.append(summary)

            human_msg = (
                f"User Query: {user_query}\n\n"
                f"Community Records:\n"
                f"{json.dumps(summaries, indent=2, ensure_ascii=False)}"
            )

            messages = [
                SystemMessage(content=COMMUNITY_FILTER_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ]

            response = self.llm.invoke(messages)
            parsed = self._parse_json_response(response.content)
            unique_ids = parsed.get("relevant_unique_ids", [])

            if not isinstance(unique_ids, list):
                return []

            return [uid for uid in unique_ids if isinstance(uid, str) and uid]

        except Exception as e:
            print(f"  ⚠️  Community ID resolution failed ({e}), skipping restriction")
            return []

    # ------------------------------------------------------------------
    # Step 5 – Source Structuring
    # ------------------------------------------------------------------

    def _structure_sources(
        self,
        evidence: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Build indexed source structure and collect community IDs.

        No community filtering happens here anymore — run() already applied
        the hard community filter (Step 4) before calling this, so every
        evidence point handed in has already passed that gate. This method
        just indexes what survived and records which community_ids are
        present, in evidence order, for grouping in Stage A/B prompts.

        Args:
            evidence: Evidence points that already survived the hard
                community filter in run().
        """
        sources: Dict[str, Dict[str, Any]] = {}
        community_ids: List[str] = []

        for i, point in enumerate(evidence):
            entities = point.get("entities", [])
            predicates = point.get("predicates", [])
            cid = point.get("community_id")
            chunk_id = point.get("chunk_id")

            sources[f"source_{i}"] = {
                "entities": [f"{j}. {e}" for j, e in enumerate(entities)],
                "predicates": [f"{j}. {p}" for j, p in enumerate(predicates)],
                "_raw_entities": entities,
                "_raw_predicates": predicates,
                # Store community_id directly on each source so downstream
                # methods can read it without positional indexing into the
                # deduplicated community_ids list.
                "_community_id": cid,
                # Store the chunk_id (Qdrant point ID or explicit source
                # ref) so a relevant source's raw text stays reachable via
                # the S3 fetch step even after entities/predicates have
                # been extracted from it.
                "_chunk_id": chunk_id,
                "_score": point.get("score"),
                # The actual quote text — shown to Stage A's source-selection
                # prompt alongside community_id. Prefer the English
                # translation when present since it's typically shorter and
                # the LLM reasons over it more reliably; fall back to the
                # original-language quote if no translation exists.
                "_quote": point.get("evidence_quote_en") or point.get("evidence_quote", ""),
            }
            if cid and cid not in community_ids:
                community_ids.append(cid)

        return sources, community_ids

    # ------------------------------------------------------------------
    # Step 6 – Stage A: Evidence Selection (numbered passages + community_id → relevant numbers)
    # ------------------------------------------------------------------

    def _select_sources(
        self, user_query: str, sources: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Ask the LLM which evidence passages are relevant, by number,
        using BOTH the quote text and the passage's community_id.

        Each source is shown as a numbered quote tagged with its
        community_id — no entity/predicate lists yet, since deciding which
        of those matter is Stage B's job and only applies to sources that
        survive this triage. Showing community_id here (not just in Step 4's
        hard filter) lets the LLM down-weight a passage whose community_id
        doesn't fit the question even though Step 4 already restricted the
        overall pool to broadly-resolved communities — Step 4 operates at
        community granularity, this operates at passage granularity.

        Args:
            user_query: Original user question
            sources: Indexed source dict from _structure_sources

        Returns:
            List of source keys (e.g. ["source_0", "source_3"]) that the
            LLM marked relevant, in their original order.
        """
        source_keys = list(sources.keys())

        numbered_passages = "\n".join(
            f"{i}. [community_id: {sources[key].get('_community_id') or 'unknown'}] "
            f"{sources[key].get('_quote', '').strip() or '(no quote text)'}"
            for i, key in enumerate(source_keys)
        )

        human_msg = (
            f"User Question: {user_query}\n\n"
            f"Passages:\n{numbered_passages}"
        )

        messages = [
            SystemMessage(content=SOURCE_SELECTION_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ]

        response = self.llm.invoke(messages)
        parsed = self._parse_json_response(response.content)

        numbers = parsed.get("relevant_passage_numbers", [])
        selected = [
            source_keys[n] for n in numbers
            if isinstance(n, int) and 0 <= n < len(source_keys)
        ]
        return selected

    # ------------------------------------------------------------------
    # Step 7 – Stage B: Entity/Predicate Filtering (grouped per community, scoped to selected sources)
    # ------------------------------------------------------------------

    def _filter_entities_predicates(
        self, user_query: str, selected_sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ask the LLM which entity/predicate indices matter, within
        sources that Stage A already selected as relevant — grouped by
        community_id so the model judges each community's items together
        instead of one flat undifferentiated list.

        Only entities/predicates belonging to pre-selected sources are ever
        shown to the model — it never has to reason about (and the prompt
        never has to enumerate) items from a source already rejected.

        Grouping by community_id addresses three things at once:
          - keeps each LLM judgment scoped to one community's items instead
            of a single giant flat list spanning every surviving community
            (size/focus)
          - makes community_id an explicit, visible signal the model can
            actually use, rather than information that existed in the
            pipeline but never reached this prompt
          - the community grouping itself gives the model a natural anchor
            for "does this entity make sense for THIS community" framing,
            which a flat list does not provide

        Args:
            user_query: Original user question
            selected_sources: Subset of the full source dict, already
                filtered down to Stage A's relevant source keys

        Returns:
            Parsed JSON dict: {"sources": {"source_0": {"entity_indices":
            [...], "predicate_indices": [...]}, ...}}
        """
        # Group source keys by community_id, preserving first-seen order so
        # the prompt's group ordering is stable run to run.
        groups: Dict[str, List[str]] = defaultdict(list)
        for key, val in selected_sources.items():
            cid = val.get("_community_id") or "unknown"
            groups[cid].append(key)

        grouped_payload: Dict[str, Dict[str, Any]] = {}
        for cid, keys in groups.items():
            grouped_payload[cid] = {
                key: {
                    "entities": selected_sources[key]["entities"],
                    "predicates": selected_sources[key]["predicates"],
                }
                for key in keys
            }

        sources_json = json.dumps(grouped_payload, indent=2, ensure_ascii=False)

        human_msg = (
            f"User Question: {user_query}\n\n"
            f"Sources (already pre-selected as relevant), grouped by community_id:\n"
            f"{sources_json}"
        )

        messages = [
            SystemMessage(content=ENTITY_PREDICATE_FILTER_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ]

        response = self.llm.invoke(messages)
        parsed = self._parse_json_response(response.content)

        # The model echoes back source keys directly under "sources" (not
        # nested under community_id) — the community grouping is presentation
        # structure for the prompt, not a change to the response schema, so
        # flatten if the model nested its response by community anyway.
        raw_sources = parsed.get("sources", {})
        flattened: Dict[str, Any] = {}
        for key, val in raw_sources.items():
            if key in selected_sources:
                flattened[key] = val
            elif isinstance(val, dict) and all(
                isinstance(v, dict) for v in val.values()
            ):
                # Model nested by community_id despite instructions —
                # unwrap one level.
                for inner_key, inner_val in val.items():
                    if inner_key in selected_sources:
                        flattened[inner_key] = inner_val

        # Stage B's schema has no "relevant" boolean per source — every
        # source given to it already passed Stage A. _restructure_by_indices
        # expects that shape historically (carrying a "relevant" flag from
        # the old single-stage prompt), so backfill relevant=True for every
        # source key present in the response, keeping that method's
        # interface stable without re-deciding relevance here.
        for source_data in flattened.values():
            source_data["relevant"] = True

        return {"sources": flattened}

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract a JSON object from LLM output — shared by both stages."""
        # Try code fence first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Bare JSON
        start = text.find("{")
        if start != -1:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return json.loads(text[start : i + 1])

        raise ValueError("Could not parse LLM response as JSON")

    # ------------------------------------------------------------------
    # Step 7 – Restructure by indices
    # ------------------------------------------------------------------

    def _restructure_by_indices(
        self,
        sources: Dict[str, Dict[str, Any]],
        filter_result: Dict[str, Any],
        community_ids: List[str],
    ) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Apply filter indices to extract actual entity/predicate names.

        Returns:
            (entity_names, predicate_names, surviving_communities,
             chunk_ids, chunk_details, raw_entities_by_community,
             raw_predicates_by_community)
            chunk_ids/chunk_details cover every source the LLM marked
            relevant — not just ones with surviving entity/predicate
            indices — since a source can be topically relevant via its raw
            text even if no individual entity/predicate index was selected.

            raw_entities_by_community / raw_predicates_by_community group
            the SAME raw names that feed entity_set/predicate_set, but keyed
            by which source's community_id they came from — an entity
            appearing in multiple communities' evidence appears in each
            group. This is what lets the orchestrator build per-community
            specs for comparison queries instead of one combined query that
            lets one community's denser results crowd out another's.
        """
        entity_set: set = set()
        predicate_set: set = set()
        surviving_communities: set = set()
        chunk_ids: List[str] = []
        chunk_details: Dict[str, Dict[str, Any]] = {}
        raw_entities_by_community: Dict[str, set] = defaultdict(set)
        raw_predicates_by_community: Dict[str, set] = defaultdict(set)

        filter_sources = filter_result.get("sources", {})

        for source_key, source_data in sources.items():
            f = filter_sources.get(source_key, {})
            if not f.get("relevant", False):
                continue

            raw_entities = source_data["_raw_entities"]
            raw_predicates = source_data["_raw_predicates"]
            cid = source_data.get("_community_id")

            for idx in f.get("entity_indices", []):
                if 0 <= idx < len(raw_entities):
                    name = raw_entities[idx]
                    entity_set.add(name)
                    if cid:
                        raw_entities_by_community[cid].add(name)

            for idx in f.get("predicate_indices", []):
                if 0 <= idx < len(raw_predicates):
                    name = raw_predicates[idx]
                    predicate_set.add(name)
                    if cid:
                        raw_predicates_by_community[cid].add(name)

            # FIX 5: read community_id stored directly on the source dict —
            # the old code used src_idx to index into the deduplicated
            # community_ids list, which is positionally unrelated to source_N.
            if cid:
                surviving_communities.add(cid)

            # NEW: carry chunk_id through for every source marked relevant,
            # regardless of whether specific entity/predicate indices were
            # also selected from it.
            chunk_id = source_data.get("_chunk_id")
            if chunk_id and chunk_id not in chunk_details:
                chunk_ids.append(chunk_id)
                chunk_details[chunk_id] = {
                    "score": source_data.get("_score"),
                    "community_id": cid,
                    # The evidence_quote/evidence_quote_en text this chunk_id
                    # came from — the only real query-relevant TEXT available
                    # before the S3 fetch happens, used by
                    # _cap_chunks_per_community for token-overlap scoring.
                    "quote": source_data.get("_quote", ""),
                }

        # If no source matched, keep all communities as fallback
        if not surviving_communities and community_ids:
            surviving_communities = set(community_ids)

        return (
            sorted(entity_set),
            sorted(predicate_set),
            sorted(surviving_communities),
            chunk_ids,
            chunk_details,
            {cid: sorted(names) for cid, names in raw_entities_by_community.items()},
            {cid: sorted(names) for cid, names in raw_predicates_by_community.items()},
        )

    def _all_from_sources(
        self,
        sources: Dict[str, Dict[str, Any]],
        community_ids: List[str],
    ) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
        """Inclusive fallback: use every entity, predicate, and chunk from evidence."""
        entity_set: set = set()
        predicate_set: set = set()
        chunk_ids: List[str] = []
        chunk_details: Dict[str, Dict[str, Any]] = {}
        raw_entities_by_community: Dict[str, set] = defaultdict(set)
        raw_predicates_by_community: Dict[str, set] = defaultdict(set)

        for val in sources.values():
            entity_set.update(val["_raw_entities"])
            predicate_set.update(val["_raw_predicates"])
            cid = val.get("_community_id")
            if cid:
                raw_entities_by_community[cid].update(val["_raw_entities"])
                raw_predicates_by_community[cid].update(val["_raw_predicates"])
            chunk_id = val.get("_chunk_id")
            if chunk_id and chunk_id not in chunk_details:
                chunk_ids.append(chunk_id)
                chunk_details[chunk_id] = {
                    "score": val.get("_score"),
                    "community_id": cid,
                    "quote": val.get("_quote", ""),
                }
        return (
            sorted(entity_set),
            sorted(predicate_set),
            community_ids,
            chunk_ids,
            chunk_details,
            {cid: sorted(names) for cid, names in raw_entities_by_community.items()},
            {cid: sorted(names) for cid, names in raw_predicates_by_community.items()},
        )

    # ------------------------------------------------------------------
    # Steps 8-9 – Canonical ID Lookup
    # ------------------------------------------------------------------

    def _lookup_entities(
        self, entity_names: List[str]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Resolve entity names to canonical IDs via Qdrant — two distinct calls.

        Call 1 (name search): embed each filtered raw entity name and
        vector-search entity_registry. Each hit's payload carries a
        canonical_id — this is "raw name X is linked to canonical_id Y."
        The raw name and the matched payload's own "name" field are NOT
        guaranteed to be the same string; they're only guaranteed to be
        semantically close (that's the whole point of vector search).

        Call 2 (canonical fetch): take the canonical_ids produced by call 1
        and fetch those exact canonical entity records directly, by
        canonical_id payload filter — no embedding, no similarity score.
        This is what actually establishes the canonical entity's own name
        as the graph/Neo4j side knows it, rather than trusting call 1's
        incidental "name" field.

        The result is an explicit raw -> (linked) -> canonical mapping:
        details[canonical_id] = {
            "name": <canonical entity's own name, from call 2>,
            "raw_names": [<every raw surface form that resolved here, from call 1>],
            "score": <best similarity score from call 1>,
        }

        Preserves the raw→canonical linkage explicitly as `raw_names`: the
        evidence_registry collection stores raw extracted entity names
        (pre-canonicalization), never canonical_ids, on its entities[] array.
        So any tool that needs to filter Qdrant evidence by entity (e.g.
        search_evidence, fetch_chunks) must use raw_names, not canonical_id
        or the canonical `name` — those only exist in Neo4j / the
        entity_registry collection, not in evidence_registry payloads.
        """
        if not entity_names:
            return [], {}

        # Call 1 — name search, raw name -> canonical_id
        search_results = self.qdrant_manager.get_canonical_entities(
            entity_names, limit=1, threshold=0.7
        )

        raw_names_by_cid: Dict[str, List[str]] = defaultdict(list)
        best_score_by_cid: Dict[str, float] = {}
        for r in search_results:
            cid = r.get("canonical_id", "")
            raw_name = r.get("query", "")
            if not cid:
                continue
            if raw_name and raw_name not in raw_names_by_cid[cid]:
                # Multiple raw surface forms can resolve to the same
                # canonical entity (e.g. "กลุ่มผู้ใช้น้ำ" and "Water User
                # Group" both → same canonical_id). Keep all of them —
                # each is a separate string that may appear in different
                # evidence_registry payloads.
                raw_names_by_cid[cid].append(raw_name)
            score = r.get("score", 0.0)
            if score > best_score_by_cid.get(cid, -1.0):
                best_score_by_cid[cid] = score

        if not raw_names_by_cid:
            return [], {}

        # Call 2 — fetch the canonical entity records themselves, by ID,
        # using exactly the canonical_ids call 1 surfaced.
        canonical_ids = list(raw_names_by_cid.keys())
        canonical_records = self.qdrant_manager.get_entities_by_canonical_ids(
            canonical_ids
        )
        canonical_name_by_cid: Dict[str, str] = {
            rec["canonical_id"]: rec.get("name", "")
            for rec in canonical_records
            if rec.get("canonical_id")
        }
        canonical_name_en_by_cid: Dict[str, str] = {
            rec["canonical_id"]: rec.get("name_en", "")
            for rec in canonical_records
            if rec.get("canonical_id")
        }

        ids: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}
        for cid in canonical_ids:
            ids.append(cid)
            details[cid] = {
                # Prefer call 2's own record; fall back to whatever the
                # search hit's payload carried if call 2 found nothing
                # (e.g. transient fetch failure) so we never silently drop
                # a canonical_id that call 1 already confirmed exists.
                "name": canonical_name_by_cid.get(cid, ""),
                # English translation of the canonical name, when the
                # registry payload actually has one — empty string
                # otherwise, never fabricated or duplicated from "name".
                "name_en": canonical_name_en_by_cid.get(cid, ""),
                "score": best_score_by_cid.get(cid, 0.0),
                "raw_names": raw_names_by_cid[cid],
            }

        return ids, details

    def _lookup_predicates(
        self, predicate_names: List[str]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Resolve predicate names to canonical IDs via Qdrant — two distinct calls.

        Same two-call linkage as _lookup_entities — see that docstring.
        Predicate evidence in Qdrant is stored as the raw extracted relation
        phrase, not the canonical Neo4j relationship type (e.g.
        evidence_registry might have "ทำให้เกิด" where Neo4j has "CAUSES"),
        and the canonical predicate's own name (which Neo4j edges connect
        through) is fetched explicitly in call 2 rather than assumed from
        call 1's search-hit payload.
        """
        if not predicate_names:
            return [], {}

        # Call 1 — name search, raw name -> canonical_id
        search_results = self.qdrant_manager.get_canonical_predicates(
            predicate_names, limit=1, threshold=0.7
        )

        raw_names_by_cid: Dict[str, List[str]] = defaultdict(list)
        best_score_by_cid: Dict[str, float] = {}
        for r in search_results:
            cid = r.get("canonical_id", "")
            raw_name = r.get("query", "")
            if not cid:
                continue
            if raw_name and raw_name not in raw_names_by_cid[cid]:
                raw_names_by_cid[cid].append(raw_name)
            score = r.get("score", 0.0)
            if score > best_score_by_cid.get(cid, -1.0):
                best_score_by_cid[cid] = score

        if not raw_names_by_cid:
            return [], {}

        # Call 2 — fetch the canonical predicate records themselves, by ID.
        canonical_ids = list(raw_names_by_cid.keys())
        canonical_records = self.qdrant_manager.get_predicates_by_canonical_ids(
            canonical_ids
        )
        canonical_name_by_cid: Dict[str, str] = {
            rec["canonical_id"]: rec.get("name", "")
            for rec in canonical_records
            if rec.get("canonical_id")
        }
        canonical_name_en_by_cid: Dict[str, str] = {
            rec["canonical_id"]: rec.get("name_en", "")
            for rec in canonical_records
            if rec.get("canonical_id")
        }

        ids: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}
        for cid in canonical_ids:
            ids.append(cid)
            details[cid] = {
                "name": canonical_name_by_cid.get(cid, ""),
                "name_en": canonical_name_en_by_cid.get(cid, ""),
                "score": best_score_by_cid.get(cid, 0.0),
                "raw_names": raw_names_by_cid[cid],
            }

        return ids, details

    # ------------------------------------------------------------------
    # Step 9.5 – Per-Community Canonical Grouping (for comparison queries)
    # ------------------------------------------------------------------

    @staticmethod
    def _group_canonical_ids_by_community(
        raw_names_by_community: Dict[str, List[str]],
        details: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Map each community's raw names to canonical_ids, for the
        orchestrator's per-community spec building.

        No new Qdrant calls — this is a local lookup against `details`
        (entity_details/predicate_details), which already maps
        canonical_id -> {"raw_names": [...]} from _lookup_entities/
        _lookup_predicates. Inverting that mapping (raw_name -> canonical_id)
        once, then looking up each community's raw names against it, is all
        that's needed.

        A canonical_id whose raw_names span multiple communities will
        appear in each of those communities' lists — this is correct, not
        a bug: an entity can genuinely be evidenced in more than one
        village's source text (e.g. a shared regional health center).

        Args:
            raw_names_by_community: community_id -> [raw names], from
                _restructure_by_indices / _all_from_sources
            details: entity_details or predicate_details — canonical_id ->
                {"raw_names": [...], ...}

        Returns:
            community_id -> sorted list of canonical_ids
        """
        if not raw_names_by_community or not details:
            return {}

        canonical_id_by_raw_name: Dict[str, str] = {}
        for canonical_id, info in details.items():
            for raw_name in info.get("raw_names", []):
                canonical_id_by_raw_name[raw_name] = canonical_id

        result: Dict[str, List[str]] = {}
        for cid, raw_names in raw_names_by_community.items():
            canonical_ids = {
                canonical_id_by_raw_name[raw_name]
                for raw_name in raw_names
                if raw_name in canonical_id_by_raw_name
            }
            if canonical_ids:
                result[cid] = sorted(canonical_ids)

        return result

    # ------------------------------------------------------------------
    # Step 9 – S3 Chunk Text Fetch
    # ------------------------------------------------------------------

    # Chunk capping — same word tokenizer style as aggregator._tokenize,
    # duplicated here (not imported) since preprocessor.py runs conceptually
    # upstream of aggregator.py and importing "downstream-to-upstream" would
    # be a backwards dependency for a few lines of regex logic.
    _CHUNK_CAP_WORD_RE = re.compile(r"[a-zA-Z0-9ก-๙]+", re.UNICODE)

    @classmethod
    def _chunk_cap_tokenize(cls, text: str) -> set:
        return {w.lower() for w in cls._CHUNK_CAP_WORD_RE.findall(text or "")}

    def _cap_chunks_per_community(
        self,
        chunk_ids: List[str],
        chunk_details: Dict[str, Dict[str, Any]],
        user_query: str,
        max_chunks_per_community: int = 5,
    ) -> List[str]:
        """Keep only the top-N most relevant chunk_ids per community,
        BEFORE the S3 fetch — avoids wasted S3 calls for chunks that would
        get cut later anyway, and keeps the aggregator's "relevant content"
        section bounded regardless of how many communities a query spans.

        Scoring mirrors aggregator._score_result's approach: token overlap
        between the user query and the chunk's evidence_quote text (carried
        into chunk_details["quote"] by _restructure_by_indices/
        _all_from_sources — see those methods), combined with the chunk's
        existing embedding similarity score from evidence search. The full
        source chunk body itself doesn't exist locally until after the S3
        fetch this method runs ahead of, so the evidence_quote excerpt is
        the best query-relevant text actually available pre-fetch.

        Args:
            chunk_ids: All filtered chunk_ids (pre-cap)
            chunk_details: chunk_id -> {"score", "community_id", "quote"}
            user_query: Original user question, tokenized and compared
                against each chunk's evidence_quote text
            max_chunks_per_community: Cap per community (default 5)

        Returns:
            Filtered chunk_ids list, capped per community, preserving each
            community's relative score order (highest first).
        """
        if not chunk_ids:
            return []

        query_tokens = self._chunk_cap_tokenize(user_query)

        def _combined_score(cid: str) -> float:
            info = chunk_details.get(cid) or {}
            embedding_score = info.get("score") or 0.0
            overlap = 0.0
            if query_tokens:
                quote_tokens = self._chunk_cap_tokenize(info.get("quote", ""))
                if quote_tokens:
                    overlap = len(query_tokens & quote_tokens) / max(len(query_tokens), 1)
            return embedding_score + overlap

        by_community: Dict[str, List[str]] = defaultdict(list)
        for cid in chunk_ids:
            community_id = (chunk_details.get(cid) or {}).get("community_id") or "unknown"
            by_community[community_id].append(cid)

        capped: List[str] = []
        for community_id, ids_in_community in by_community.items():
            ranked = sorted(ids_in_community, key=_combined_score, reverse=True)
            kept = ranked[:max_chunks_per_community]
            if len(ranked) > max_chunks_per_community:
                print(f"  ✂️  Capping chunks for {community_id}: "
                      f"{len(ranked)} → {len(kept)}")
            capped.extend(kept)

        return capped

    def _fetch_s3_chunk_texts(
        self,
        chunk_ids: List[str],
        chunk_details: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch original source chunk text from S3 for every filtered chunk.

        This is the step your stated strategy calls out explicitly: "from
        the community_id and chunk_id from filtered evidence points, query
        s3 (not qdrant) for chunks" at
        cmu_ci_documents/{community_id}/chunk_{chunk_id:03d}.txt.

        chunk_details already carries {"score", "community_id"} per
        chunk_id from _restructure_by_indices / _all_from_sources — the
        community_id there is exactly the S3 key prefix, and chunk_id is
        the same integer the evidence_registry payload stores in its
        "chunk_id" field (see qdrant_tools.get_evidence_by_ids /
        search_evidence chunk_id resolution) and that neo4j_graph_builder.py
        writes onto relationship properties. No ID translation is needed —
        this just builds (community_id, chunk_id) refs and calls
        chunk_tools.fetch_s3_chunks directly, in Python, no LLM involved.

        Chunks missing a community_id (e.g. a fallback path that produced
        chunk_ids without details) are skipped rather than guessed at.

        Callers are expected to pass an already-capped chunk_ids list (see
        _cap_chunks_per_community) — this method itself fetches whatever
        list it's given, with no further capping, so capping always
        happens once, in one place, before any S3 call.

        Args:
            chunk_ids: Filtered (and capped) chunk IDs from Stage A/B (or
                the inclusive fallback), in priority order.
            chunk_details: chunk_id -> {"score", "community_id"}

        Returns:
            chunk_id -> {"community_id", "text", "s3_key", "error"?}
        """
        if not chunk_ids:
            return {}

        refs = []
        skipped: List[str] = []
        for cid in chunk_ids:
            community_id = (chunk_details.get(cid) or {}).get("community_id")
            if not community_id:
                skipped.append(cid)
                continue
            try:
                int_chunk_id = int(cid)
            except (TypeError, ValueError):
                skipped.append(cid)
                continue
            refs.append({"community_id": community_id, "chunk_id": int_chunk_id})

        if skipped:
            print(f"  ⚠️  Skipping S3 fetch for {len(skipped)} chunk(s) without "
                  f"a resolvable community_id/integer chunk_id")

        if not refs:
            return {}

        try:
            fetched = fetch_s3_chunks(refs)
        except Exception as e:
            print(f"  ⚠️  S3 chunk fetch failed entirely ({e})")
            return {}

        if fetched and "error" in fetched[0] and "community_id" not in fetched[0]:
            # Top-level failure (missing boto3/creds) — applies to the whole
            # batch, not just one ref.
            print(f"  ⚠️  S3 chunk fetch unavailable: {fetched[0]['error']}")
            return {}

        # fetch_s3_chunks deduplicates internally by (community_id, chunk_id),
        # so its output list can be shorter than `refs` — match by key
        # rather than position to stay correct regardless of dedup.
        by_ref: Dict[Tuple[str, int], Dict[str, Any]] = {
            (r.get("community_id"), r.get("chunk_id")): r for r in fetched
        }

        texts: Dict[str, Dict[str, Any]] = {}
        for ref in refs:
            key = (ref["community_id"], ref["chunk_id"])
            result = by_ref.get(key)
            if result is None:
                continue
            original_id = str(ref["chunk_id"])
            texts[original_id] = {
                "community_id": result.get("community_id"),
                "text": result.get("text", ""),
                "s3_key": result.get("s3_key"),
                "error": result.get("error"),
            }

        return texts

    # ------------------------------------------------------------------
    # Step 10 – Build Output
    # ------------------------------------------------------------------

    def _build_output(
        self,
        user_query: str,
        expanded_query: str,
        entity_ids: List[str],
        entity_details: Dict[str, Dict[str, Any]],
        predicate_ids: List[str],
        predicate_details: Dict[str, Dict[str, Any]],
        community_ids: List[str],
        community_labels: Optional[Dict[str, str]] = None,
        chunk_ids: Optional[List[str]] = None,
        chunk_details: Optional[Dict[str, Dict[str, Any]]] = None,
        chunk_texts: Optional[Dict[str, Dict[str, Any]]] = None,
        entities_by_community: Optional[Dict[str, List[str]]] = None,
        predicates_by_community: Optional[Dict[str, List[str]]] = None,
        all_filtered_raw_entities: Optional[List[str]] = None,
        all_filtered_raw_predicates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assemble final output with query type classification.

        raw_entity_names / raw_predicate_names are the union of:
          (a) every raw surface form that resolved to a canonical_id, and
          (b) every raw name that survived LLM filtering even if it never
              resolved to a canonical_id (e.g. scored below the 0.7
              similarity threshold).
        This union — NOT entity_ids/predicate_ids — is what tools must use
        to filter Qdrant's evidence_registry collection, since that
        collection's entities[]/predicates[] payload arrays are populated
        from raw pre-canonicalization triple text (see triple_refiner.py's
        evidence grouping step) and never carry canonical_id on those fields.
        Dropping (b) would silently lose evidence filter terms whenever
        canonicalization fails, even though the raw text is still exactly
        what's stored in Qdrant.

        entity_ids/entity_details together ARE the raw→canonical linkage:
        each canonical_id's detail dict carries raw_names — every raw
        surface form from evidence_registry that resolved to it. This is
        what the aggregator needs for its "raw name -> (linked) ->
        canonical names" section.

        entities_by_community / predicates_by_community are what let the
        orchestrator build PER-COMMUNITY worker_specs for comparison
        queries (2+ communities) instead of one combined query whose
        shared LIMIT lets one community's denser results crowd out
        another's.
        """
        query_type, needs_pathing, needs_community = self._classify_query(
            user_query, entity_ids, community_ids
        )

        raw_entity_names = sorted({
            name
            for d in (entity_details or {}).values()
            for name in d.get("raw_names", [])
        } | set(all_filtered_raw_entities or []))

        raw_predicate_names = sorted({
            name
            for d in (predicate_details or {}).values()
            for name in d.get("raw_names", [])
        } | set(all_filtered_raw_predicates or []))

        return {
            "expanded_query": expanded_query,
            "entity_ids": entity_ids,
            "predicate_ids": predicate_ids,
            "community_ids": community_ids,
            "community_labels": community_labels or {},
            "entity_details": entity_details,
            "predicate_details": predicate_details,
            "entities_by_community": entities_by_community or {},
            "predicates_by_community": predicates_by_community or {},
            "raw_entity_names": raw_entity_names,
            "raw_predicate_names": raw_predicate_names,
            "chunk_ids": chunk_ids or [],
            "chunk_details": chunk_details or {},
            "chunk_texts": chunk_texts or {},
            "query_type": query_type,
            "needs_pathing": needs_pathing,
            "needs_community": needs_community,
        }

    def _classify_query(
        self,
        user_query: str,
        entity_ids: List[str],
        community_ids: List[str],
    ) -> Tuple[str, bool, bool]:
        """Classify query type using heuristics (no LLM call)."""
        is_community_query = bool(_COMMUNITY_KEYWORDS.search(user_query))
        needs_community = is_community_query and len(community_ids) > 0
        needs_pathing = len(entity_ids) >= 2

        if needs_community:
            query_type = "community"
        elif needs_pathing:
            query_type = "pathing"
        elif len(entity_ids) == 1:
            query_type = "exploration"
        else:
            query_type = "general"

        return query_type, needs_pathing, needs_community

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(expanded_query: str = "") -> Dict[str, Any]:
        return {
            "expanded_query": expanded_query,
            "entity_ids": [],
            "predicate_ids": [],
            "community_ids": [],
            "community_labels": {},
            "entity_details": {},
            "predicate_details": {},
            "entities_by_community": {},
            "predicates_by_community": {},
            "raw_entity_names": [],
            "raw_predicate_names": [],
            "chunk_ids": [],
            "chunk_details": {},
            "chunk_texts": {},
            "query_type": "general",
            "needs_pathing": False,
            "needs_community": False,
        }