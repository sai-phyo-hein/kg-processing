"""PreProcessor agent for multi-agent reasoning system.

Runs a deterministic 9-step pipeline before the orchestrator:
1. Query expansion (fast LLM)
2-3. Evidence search (dense + sparse hybrid via QdrantToolsManager)
4. Source structuring (indexed entity/predicate lists per evidence point)
5. LLM filtering (return index-only selection)
6. Restructure lists by LLM-selected indices
7-8. Canonical ID lookup (embed + query entity/predicate registries)
9. Assemble output with query type classification
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from kg_extractor.utils.model_setup import get_reasoning_llm, PREPROCESSING_MODEL
from kg_reasoning.agents.tools.qdrant_tools import _get_manager
from kg_reasoning.prompts.preprocessor_prompts import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    SOURCE_FILTER_SYSTEM_PROMPT,
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
        self.llm = get_reasoning_llm(model=llm_model, temperature=0.1)
        self.qdrant_manager = _get_manager()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, user_query: str) -> Dict[str, Any]:
        """Execute the full 9-step pipeline.

        Returns a dict with keys:
            expanded_query, entity_ids, predicate_ids, community_ids,
            entity_details, predicate_details, query_type,
            needs_pathing, needs_community
        """
        # Step 1 – expand
        expanded = self._expand_query(user_query)
        print(f"  📝 Expanded query: {expanded[:120]}...")

        # Steps 2-3 – evidence search
        evidence = self._search_evidence(expanded)
        print(f"  🔎 Evidence points: {len(evidence)}")
        if not evidence:
            return self._empty_result(expanded)

        # Step 4 – structure sources
        sources, community_ids = self._structure_sources(evidence)
        print(f"  📋 Structured {len(sources)} sources, {len(community_ids)} communities")

        # Steps 5-6 – filter with LLM
        try:
            filter_result = self._filter_with_llm(user_query, sources)
            filtered_entities, filtered_predicates, filtered_communities = (
                self._restructure_by_indices(sources, filter_result, community_ids)
            )
            print(f"  🎯 Filtered: {len(filtered_entities)} entities, {len(filtered_predicates)} predicates")
        except Exception as e:
            # Inclusive fallback: use everything from evidence
            print(f"  ⚠️  LLM filter failed ({e}), using all evidence items")
            filtered_entities, filtered_predicates, filtered_communities = (
                self._all_from_sources(sources, community_ids)
            )

        if not filtered_entities and not filtered_predicates:
            print("  ⚠️  No entities or predicates after filtering")
            return self._empty_result(expanded)

        # Steps 7-8 – canonical ID lookup (parallel)
        with ThreadPoolExecutor(max_workers=2) as pool:
            entity_future = pool.submit(self._lookup_entities, filtered_entities)
            predicate_future = pool.submit(self._lookup_predicates, filtered_predicates)
            entity_ids, entity_details = entity_future.result()
            predicate_ids, predicate_details = predicate_future.result()
        print(f"  🔗 Resolved: {len(entity_ids)} entity IDs, {len(predicate_ids)} predicate IDs")

        # Step 9 – assemble output
        return self._build_output(
            user_query=user_query,
            expanded_query=expanded,
            entity_ids=entity_ids,
            entity_details=entity_details,
            predicate_ids=predicate_ids,
            predicate_details=predicate_details,
            community_ids=filtered_communities,
        )

    # ------------------------------------------------------------------
    # Step 1 – Query Expansion
    # ------------------------------------------------------------------

    def _expand_query(self, user_query: str) -> str:
        messages = [
            SystemMessage(content=QUERY_EXPANSION_SYSTEM_PROMPT),
            HumanMessage(content=user_query),
        ]
        response = self.llm.invoke(messages)
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
    # Step 4 – Source Structuring
    # ------------------------------------------------------------------

    def _structure_sources(
        self, evidence: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """Build indexed source structure and collect community IDs."""
        sources: Dict[str, Dict[str, Any]] = {}
        community_ids: List[str] = []

        for i, point in enumerate(evidence):
            entities = point.get("entities", [])
            predicates = point.get("predicates", [])
            cid = point.get("community_id")

            sources[f"source_{i}"] = {
                "entities": [f"{j}. {e}" for j, e in enumerate(entities)],
                "predicates": [f"{j}. {p}" for j, p in enumerate(predicates)],
                "_raw_entities": entities,
                "_raw_predicates": predicates,
                # FIX 5: store community_id directly on each source so
                # _restructure_by_indices can read it without positional
                # indexing into the deduplicated community_ids list.
                "_community_id": cid,
            }
            if cid and cid not in community_ids:
                community_ids.append(cid)

        return sources, community_ids

    # ------------------------------------------------------------------
    # Step 5 – LLM Filtering
    # ------------------------------------------------------------------

    def _filter_with_llm(
        self, user_query: str, sources: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ask LLM to select relevant source/entity/predicate indices."""
        # Strip internal keys for the prompt
        clean_sources = {}
        for key, val in sources.items():
            clean_sources[key] = {
                "entities": val["entities"],
                "predicates": val["predicates"],
            }

        sources_json = json.dumps(clean_sources, indent=2, ensure_ascii=False)

        human_msg = (
            f"User Question: {user_query}\n\n"
            f"Sources:\n{sources_json}"
        )

        messages = [
            SystemMessage(content=SOURCE_FILTER_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ]

        response = self.llm.invoke(messages)
        return self._parse_filter_response(response.content)

    def _parse_filter_response(self, text: str) -> Dict[str, Any]:
        """Extract the JSON filter result from LLM output."""
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

        raise ValueError("Could not parse filter response as JSON")

    # ------------------------------------------------------------------
    # Step 6 – Restructure by indices
    # ------------------------------------------------------------------

    def _restructure_by_indices(
        self,
        sources: Dict[str, Dict[str, Any]],
        filter_result: Dict[str, Any],
        community_ids: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Apply filter indices to extract actual entity/predicate names."""
        entity_set: set = set()
        predicate_set: set = set()
        surviving_communities: set = set()

        filter_sources = filter_result.get("sources", {})

        for source_key, source_data in sources.items():
            f = filter_sources.get(source_key, {})
            if not f.get("relevant", False):
                continue

            raw_entities = source_data["_raw_entities"]
            raw_predicates = source_data["_raw_predicates"]

            for idx in f.get("entity_indices", []):
                if 0 <= idx < len(raw_entities):
                    entity_set.add(raw_entities[idx])

            for idx in f.get("predicate_indices", []):
                if 0 <= idx < len(raw_predicates):
                    predicate_set.add(raw_predicates[idx])

            # FIX 5: read community_id stored directly on the source dict —
            # the old code used src_idx to index into the deduplicated
            # community_ids list, which is positionally unrelated to source_N.
            cid = source_data.get("_community_id")
            if cid:
                surviving_communities.add(cid)

        # If no source matched, keep all communities as fallback
        if not surviving_communities and community_ids:
            surviving_communities = set(community_ids)

        return (
            sorted(entity_set),
            sorted(predicate_set),
            sorted(surviving_communities),
        )

    def _all_from_sources(
        self,
        sources: Dict[str, Dict[str, Any]],
        community_ids: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Inclusive fallback: use every entity and predicate from evidence."""
        entity_set: set = set()
        predicate_set: set = set()
        for val in sources.values():
            entity_set.update(val["_raw_entities"])
            predicate_set.update(val["_raw_predicates"])
        return sorted(entity_set), sorted(predicate_set), community_ids

    # ------------------------------------------------------------------
    # Steps 7-8 – Canonical ID Lookup
    # ------------------------------------------------------------------

    def _lookup_entities(
        self, entity_names: List[str]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Resolve entity names to canonical IDs via Qdrant."""
        if not entity_names:
            return [], {}

        results = self.qdrant_manager.get_canonical_entities(
            entity_names, limit=1, threshold=0.7
        )

        ids: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        for r in results:
            cid = r.get("canonical_id", "")
            if cid and cid not in details:
                ids.append(cid)
                details[cid] = {
                    "name": r.get("name", ""),
                    "score": r.get("score", 0.0),
                    "query": r.get("query", ""),
                }

        return ids, details

    def _lookup_predicates(
        self, predicate_names: List[str]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Resolve predicate names to canonical IDs via Qdrant."""
        if not predicate_names:
            return [], {}

        results = self.qdrant_manager.get_canonical_predicates(
            predicate_names, limit=1, threshold=0.7
        )

        ids: List[str] = []
        details: Dict[str, Dict[str, Any]] = {}

        for r in results:
            cid = r.get("canonical_id", "")
            if cid and cid not in details:
                ids.append(cid)
                details[cid] = {
                    "name": r.get("name", ""),
                    "score": r.get("score", 0.0),
                    "query": r.get("query", ""),
                }

        return ids, details

    # ------------------------------------------------------------------
    # Step 9 – Build Output
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
    ) -> Dict[str, Any]:
        """Assemble final output with query type classification."""
        query_type, needs_pathing, needs_community = self._classify_query(
            user_query, entity_ids, community_ids
        )

        return {
            "expanded_query": expanded_query,
            "entity_ids": entity_ids,
            "predicate_ids": predicate_ids,
            "community_ids": community_ids,
            "entity_details": entity_details,
            "predicate_details": predicate_details,
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
            "entity_details": {},
            "predicate_details": {},
            "query_type": "general",
            "needs_pathing": False,
            "needs_community": False,
        }