"""Orchestrator agent for multi-agent reasoning system.

STRATEGY (exact, per spec): based on the preprocessor's filtered entities
and predicates, build graph query strategies using ONLY three tools:
  - graph_hop          ("neighborhoods" — 1-hop expansion from entities)
  - find_paths         ("path between"  — shortest paths between 2+ entities)
  - community_explore  ("large communities" — fallback when there are NO
                         filtered entities AND NO filtered predicates at all)

Branch selection is DETERMINISTIC Python, not an LLM judgment call — it's a
strict count-based rule:
  - 0 or 1 filtered entity            -> graph_hop only (falls through to
                                          community_explore if there are also
                                          no predicates at all to anchor on)
  - 2+ filtered entities              -> BOTH graph_hop AND find_paths, as
                                          two separate worker_specs
  - 0 filtered entities AND 0 filtered predicates -> community_explore only

Once the branch(es) are chosen, a SINGLE LLM call decides PARAMETERS within
those branches (graph_hop depth, find_paths depth/limit, which predicate_ids
to use as a relationship-type filter, limits) — the LLM is never asked to
pick the tool, only to tune it. This keeps the prompt narrow and decidable
instead of the previous 9-rule free-for-all spanning 9 different tools.

search_evidence, direct, fetch_chunks, surround_chunks, fetch_s3_chunks, and
surround_s3_chunks are REMOVED from the orchestrator's vocabulary entirely —
not deprioritized, not optional fallbacks, gone. The preprocessor's S3 fetch
step (see preprocessor.py Step 9) already retrieves chunk text directly, so
the orchestrator has no chunk-level retrieval to plan; its only job is graph
topology strategy.

The original ReAct-based analyze_and_plan (no-preprocessor fallback) is kept
for when the preprocessor fails entirely — it still needs to call Qdrant
lookup tools itself in that case, since there are no filtered entities yet
to branch on.
"""

from typing import Any, Dict, List, Optional, Tuple

from kg_extractor.utils.model_setup import REASONING_PROVIDER, ORCHESTRATOR_MODEL, get_reasoning_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from kg_reasoning.agents.tools.qdrant_tools import (
    get_canonical_entities,
    get_canonical_predicates,
    get_community_metadata,
    get_labels_by_group,
    get_entities_by_labels,
    get_all_community_ids,
)
from kg_reasoning.agents.tools.neo4j_tools import (
    get_top_connected_nodes,
    get_community_ids_from_relationships,
)


# Fixed tool vocabulary the orchestrator may emit specs for — exactly the
# three tools the strategy calls for. Keep this in sync with TOOL_MAP in
# worker.py, which only defines executors for these three (plus their
# legacy "approach"-string aliases for the ReAct fallback path).
VALID_TOOLS = {
    "graph_hop",          # neighborhoods — 1-hop expansion from entities
    "find_paths",         # path between — shortest paths between 2+ entities
    "community_explore",  # large communities — fallback, no entities/predicates
}


ORCHESTRATOR_SYSTEM_PROMPT = """You are an Orchestrator Agent for a knowledge graph reasoning system.

Your role is to:
1. Analyze the user's question to identify key entities, relationships, and concepts
2. Use Qdrant tools to find canonical IDs for entities and predicates
3. Check community metadata to understand how the graph is organized
4. Plan strategies that could answer the question, using ONLY the three approaches below

## Available Tools

You have access to these tools:
- get_canonical_entities: Find canonical entity IDs from entity_registry
- get_canonical_predicates: Find canonical predicate IDs from predicate_registry
- get_community_metadata: Get information about graph communities
- get_labels_by_group: Get labels filtered by label group (e.g., "Community", "Person")
- get_entities_by_labels: Get entities filtered by specific label names
- get_all_community_ids: Get list of all available community IDs in the graph
- get_top_connected_nodes: Get the 20 most connected nodes in the graph
- get_community_ids_from_relationships: Get actual community IDs from Neo4j relationship properties

## Qdrant Payload Schema

Both entity_registry and predicate_registry points have the following payload fields:
- **canonical_id** (uuid): Use this UUID as the stable identifier when building strategies
- **name** (keyword): Human-readable canonical name; use this for display and keyword filtering
- **evidence_quote** / **evidence_quote_en**: the original source sentence (and its English
  translation) the entity/predicate was extracted from — already present on the SAME point,
  surfaced under the `metadata` field of get_canonical_entities/get_canonical_predicates
  results. There is no separate evidence-search tool and none is needed: if you need to see
  the source text behind a canonical entity/predicate, read it directly off the result you
  already have from those two tools, not a fresh search.

When interpreting tool results, always extract `canonical_id` (not any other ID field) for use
in strategy `canonical_ids`, and use `name` when you need a human-readable label.

## Query Strategy Planning

For each strategy, specify:
- **name**: Short descriptive name (e.g., "entity_neighborhood", "entity_path_search")
- **description**: What this strategy aims to find
- **approach**: MUST be exactly one of: "expansion" (neighborhoods), "path" (path between
  entities), or "community_exploration" (large communities fallback) — no other value is valid
- **canonical_ids**: Relevant entity/predicate canonical IDs to use
- **community_ids**: Relevant community IDs to filter by (if applicable)
- **parameters**: depth, limit, and predicate_properties (see below) — NOT relationship_types
  filters or anything resembling a direct-lookup strategy

## Strategy Types — EXACTLY These Three, Nothing Else

Use this strict rule based on how many canonical entity IDs you found:
1. **0 entities found AND 0 predicates found**: emit ONLY a community_exploration strategy
   (approach: "community_exploration") — query all relationships filtered by community_id,
   since there is no entity/predicate anchor to narrow the search.
2. **Exactly 1 entity found**: emit ONLY an expansion strategy (approach: "expansion") —
   1-hop neighborhood exploration from that entity.
3. **2+ entities found**: emit BOTH an expansion strategy (approach: "expansion") AND a path
   strategy (approach: "path") between the entities — two separate strategies.

Do NOT emit a "direct" triple-lookup strategy, a predicate-only strategy, or any strategy
whose approach is not one of "expansion", "path", "community_exploration". This is a strict
restriction, not a preference — strategies of any other shape will be discarded downstream.

## Predicate Properties

Relationships in this graph carry attributes inconsistently — causal weight/strength,
temporal markers, or other attributes on some edges and not others. Whatever you set, the
worker returns ALL properties that actually exist on each matched edge regardless — but
include a `predicate_properties` list in `parameters` naming which property KINDS would be
meaningful for this specific question (e.g. "weight", "causal_strength", "temporal_link",
"date", "duration" — whatever the question implies matters), so the aggregator knows which
returned properties are signal versus noise for this question. Leave it empty only if the
question is purely about connectivity, with no notion of strength or timing.

## Community-Based Query Filtering

**IMPORTANT**: If the user's query is community-based (mentions specific village, "moo"/"หมู่", sub-district, or location):

1. **Get available community IDs**: FIRST call `get_community_ids_from_relationships()` to see what community IDs exist in Neo4j relationships
2. **Match the query to a community_id**: Match the user's query to available community IDs (e.g., "หมู่ 1" might match "หมู่ 1_village" or similar)
3. Include the matched `community_id` in every strategy's `community_ids` list, regardless of which of the three approaches you're using

**Neo4j Relationship Structure**:
- Relationships (predicates) have a `community_id` property
- Example: `(subject)-[r]->(object)` where `r.community_id = "หมู่ 1_village"`
- Worker agents will filter relationships using: `WHERE r.community_id IN $community_ids`

## Fallback When No Entities or Predicates Are Found

If get_canonical_entities and get_canonical_predicates return no matches, and there is also
no community_id to filter by: call get_top_connected_nodes to retrieve the 20 most connected
nodes, treat their canonical_ids as the seed entities, and apply rule 2 or 3 above based on
how many you end up with. Do NOT emit a bare "fallback_query" with empty canonical_ids and no
community_ids.

## Important Notes

- Edges in the graph can be differentiated by community_id in their metadata
- Use community_metadata_registry to understand which communities might be relevant
- Apply the strict entity-count rule above — do not add extra strategies beyond what it calls for
- Each strategy should be distinct and offer a different perspective

Output your final plan as a structured JSON object with a "strategies" list."""


# ---------------------------------------------------------------------------
# New context-aware prompt: emits worker_specs, a fixed tool vocabulary,
# instead of free-form "approach" strings.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# New prompt: branch selection (which tool(s) fire) is decided in Python —
# see _select_branches — BEFORE this prompt runs. This prompt only tunes
# PARAMETERS within the already-chosen branch(es): depth, limit, which
# predicate_ids to filter by, and which predicate PROPERTIES to look for
# on each edge (causal weight, temporal marker, or any other attribute the
# question implies matters) so the aggregator can surface them later.
# ---------------------------------------------------------------------------
ORCHESTRATOR_SPEC_PROMPT = """You are an Orchestrator Agent for a knowledge graph reasoning system.

A pre-processor has already resolved canonical entity IDs and predicate IDs from the
evidence, GROUPED BY COMMUNITY, and a branch-selection step (pure Python, already done —
not your job) has already decided, FOR EACH COMMUNITY INDEPENDENTLY, which tool(s) will
run: graph_hop ("neighborhoods"), find_paths ("path between"), and/or community_explore
("large communities" fallback). You will NOT see any other tool, and you must NOT invent
one.

Your ONLY job here is to tune PARAMETERS for the worker_spec(s) already chosen for you —
you do not choose which tools run, and you do not choose which community a spec applies
to; both decisions are final and listed below per community section.

## Why Communities Are Separated

When a question compares 2+ communities (e.g. "compare village 1 and village 2"), each
community got its OWN branch decision using ONLY that community's own entities/predicates
— never a combined pool. This is deliberate: it stops one community's denser graph
connections from crowding out another's in a shared query. You must preserve this
separation. A spec built for community A's branch uses ONLY community A's entity_ids/
predicate_ids and ONLY community A's ID in its own "community_ids" field — never blend
two communities into one spec, even when they share the same branch name.

## Two Parallel ID Systems — Do Not Mix Them

- **Canonical (refined) names/IDs** — live in Neo4j and the entity_registry/predicate_registry
  Qdrant collections. Example: canonical name "Water User Group", predicate "IMPACT_ON".
- **Raw (extracted) names** — live ONLY in evidence_registry, not used by any tool you can
  emit specs for (graph_hop/find_paths/community_explore are Neo4j-only, canonical IDs only).

You will only ever see canonical entity_ids/predicate_ids in context — never raw names —
since none of your three tools touch evidence_registry.

## Pre-Processed Context (already resolved — do not re-resolve)

For each community listed below, you'll see:
- **Entity Canonical IDs**: UUIDs with canonical names and match scores, scoped to THIS community
- **Predicate Canonical IDs**: same, scoped to THIS community — pay attention to the
  canonical NAME of each predicate, since it often hints at what kind of property that
  relationship type tends to carry (e.g. "IMPACT_ON"/"CAUSES" plausibly carries a
  causal-strength/weight property; a predicate involving dates/duration plausibly carries
  a temporal property)
- **Branches already selected for THIS community**: the fixed list of tool(s) you must
  emit exactly one worker_spec for, each scoped to ONLY this community's IDs above

## Parameters You Decide, Per (Community, Branch) Pair

### graph_hop ("neighborhoods")
- `depth`: 1 or 2. Use 2 only if the question seems to need indirect/second-order
  connections, not just "what's directly connected to X"
- `limit`: how many relationships to return (default 50; raise for broad exploratory
  questions, lower for narrow ones). When comparing communities, keep limits comparable
  across communities unless the question itself implies one needs deeper coverage.
- `relationship_types`: leave empty unless predicate_ids strongly suggest the question
  only cares about specific relationship types

### find_paths ("path between")
- `depth`: max path length, 2-4. Use the question's apparent complexity — "how is X
  connected to Y" is usually depth 3; "is X directly linked to Y" can be depth 2
- `limit`: how many distinct paths to return (default 20)

### community_explore ("large communities" fallback)
- `limit`: how many relationships to return (default 100, since this branch only fires
  when there's no entity/predicate anchor to narrow the search for that community)

## Predicate Properties — Required For Every Spec

Relationships in this graph carry attributes inconsistently: some edges have a causal
weight/strength property, some have a temporal marker (when something happened or how
long it lasted), some have neither, some may have other attributes entirely. The worker
will return WHATEVER properties actually exist on each matched edge regardless of what
you write here — but you must still name, in a `predicate_properties` list inside
`parameters`, which property KINDS this question's answer would benefit from, so the
aggregator downstream knows which returned properties are meaningful signal for THIS
question versus incidental noise. This applies per spec, the same way regardless of which
community the spec belongs to — the question's own intent (not the community) determines
what property kinds matter.

Think about the question itself: does it ask "how strongly does X affect Y" (→ name a
causal/weight-style property), "when did X happen" or "how long did X last" (→ name a
temporal-style property), or something else entirely attribute-based? Name your best
guess at the property key(s) a relevant edge would use — you do not know the exact field
names in advance, so list plausible ones (e.g. "weight", "causal_strength", "temporal_link",
"date", "duration" — whatever fits the question) rather than leaving this empty by default.
Only leave `predicate_properties` empty if the question is purely about which things are
connected, with no notion of strength, timing, or other attribute mattering at all.

## Output Format

Return ONLY this JSON object, no prose, no markdown fence labels beyond the fence itself.
Emit EXACTLY one worker_spec per (community, branch) pair listed across all community
sections above — do not add specs for pairs not listed, do not omit specs for pairs that
are listed, do not merge two communities' specs into one even for the same branch name.

```json
{{
  "query_type": "<from context>",
  "entities_found": ["<entity UUIDs from context>"],
  "predicates_found": ["<predicate UUIDs from context>"],
  "communities_identified": ["<community IDs from context>"],
  "worker_specs": [
    {{
      "tool": "graph_hop",
      "name": "entity_neighborhood_<community_id>",
      "entity_ids": ["<uuid from THIS community's section only>"],
      "predicate_ids": [],
      "chunk_ids": [],
      "community_ids": ["<THIS community's ID only — exactly one>"],
      "parameters": {{
        "depth": 1,
        "limit": 50,
        "predicate_properties": ["causal_strength", "temporal_link"]
      }}
    }}
  ]
}}
```

Every worker_spec MUST have a "tool" field whose value is one of: graph_hop, find_paths,
community_explore — exactly matching one of the branches already selected for that spec's
community, no other tool name is valid. Every worker_spec's "community_ids" field must
contain exactly the one community_id its entity_ids/predicate_ids were scoped to (or be
empty only for the single/ungrouped case where no community section showed an ID). Use
actual UUIDs from the context block, never placeholder text. Every worker_spec's
`parameters` MUST include a `predicate_properties` key (an array, possibly empty per the
rule above)."""




class OrchestratorAgent:
    """Orchestrator agent for query analysis and strategy planning."""

    def __init__(
        self,
        llm_provider: str = REASONING_PROVIDER,
        llm_model: str = ORCHESTRATOR_MODEL,
        temperature: float = 0.1,
    ):
        """Initialize the orchestrator agent.

        Args:
            llm_provider: LLM provider (supports: openai, openrouter, groq, nvidia)
            llm_model: Model name
            temperature: LLM temperature for response generation
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm = get_reasoning_llm(model=llm_model, temperature=temperature)

        # Tools for the ReAct fallback path only (no-preprocessor case).
        # No separate evidence-search/lookup tool: evidence_quote and
        # evidence_quote_en already live directly on each entity_registry/
        # predicate_registry point's own payload, surfaced under
        # result["metadata"] by get_canonical_entities/get_canonical_predicates
        # — there is no other place to "look back" the evidence from, and
        # no separate evidence_registry vector search is needed or correct
        # here.
        self.tools = [
            get_canonical_entities,
            get_canonical_predicates,
            get_community_metadata,
            get_labels_by_group,
            get_entities_by_labels,
            get_all_community_ids,
            get_top_connected_nodes,
            get_community_ids_from_relationships,
        ]
        self.agent = create_react_agent(self.llm, self.tools)

    # ------------------------------------------------------------------
    # NEW primary path: deterministic branch selection (Python) + a single
    # LLM call that only tunes parameters within the chosen branch(es).
    # ------------------------------------------------------------------

    @staticmethod
    def _select_branches(entity_ids: List[str], predicate_ids: List[str]) -> List[str]:
        """Decide which tool(s) fire for ONE set of entities/predicates —
        pure Python, no LLM judgment call.

        Strict rule, exactly as specified:
          - 0 filtered entities AND 0 filtered predicates -> community_explore only
          - 1 filtered entity (any predicate count)        -> graph_hop only
          - 2+ filtered entities                            -> graph_hop AND find_paths,
                                                                as two separate branches

        Returns the ordered list of tool names to emit specs for.
        """
        if not entity_ids and not predicate_ids:
            return ["community_explore"]
        if len(entity_ids) >= 2:
            return ["graph_hop", "find_paths"]
        return ["graph_hop"]

    @staticmethod
    def _plan_per_community(
        community_ids: List[str],
        entities_by_community: Dict[str, List[str]],
        predicates_by_community: Dict[str, List[str]],
        entity_ids: List[str],
        predicate_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Decide branches for EACH community independently, so a
        comparison query can't let one community's denser results crowd
        out another's in a shared query/limit.

        For a single-community query (len(community_ids) <= 1), this
        collapses to the original behavior: one "community" entry using
        the full flat entity_ids/predicate_ids, with community_id=None so
        downstream spec-building doesn't add a redundant community filter
        beyond what's already in state["preprocessor_community_ids"].

        For 2+ communities, each community gets its OWN entity_ids slice
        (from entities_by_community — see preprocessor.py's
        _group_canonical_ids_by_community) and its own _select_branches
        call, so e.g. village 1 with 10 entities gets graph_hop+find_paths
        while village 2 with 0 entities falls back to community_explore —
        instead of both villages being forced through one combined spec
        whose branch choice and shared LIMIT are dictated by whichever
        village happens to have more entities.

        Returns a list of {"community_id", "entity_ids", "predicate_ids",
        "branches"} dicts, one per community (or one total for the
        single/no-community case).
        """
        if len(community_ids) <= 1:
            return [{
                "community_id": community_ids[0] if community_ids else None,
                "entity_ids": entity_ids,
                "predicate_ids": predicate_ids,
                "branches": OrchestratorAgent._select_branches(entity_ids, predicate_ids),
            }]

        plans = []
        for cid in community_ids:
            cid_entities = entities_by_community.get(cid, [])
            cid_predicates = predicates_by_community.get(cid, [])
            plans.append({
                "community_id": cid,
                "entity_ids": cid_entities,
                "predicate_ids": cid_predicates,
                "branches": OrchestratorAgent._select_branches(cid_entities, cid_predicates),
            })
        return plans

    def analyze_and_plan_with_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan worker_specs using pre-resolved canonical IDs.

        Branch selection (which of graph_hop / find_paths / community_explore
        fire) is decided in Python via _plan_per_community BEFORE the LLM is
        ever called — it's a strict count-based rule, not a judgment call,
        applied independently to EACH community so a comparison query gets
        balanced coverage rather than one combined query per tool. The
        single LLM call that follows only tunes parameters (depth, limit,
        predicate_ids filter, predicate_properties) within those
        already-chosen (community, branch) pairs; it cannot add, remove,
        or reassign a branch to a different community.

        Args:
            state: Workflow state containing pre-processor results

        Returns:
            Dictionary containing worker_specs and metadata
        """
        entity_ids = state.get("preprocessor_entity_ids", [])
        predicate_ids = state.get("preprocessor_predicate_ids", [])
        community_ids = state.get("preprocessor_community_ids", [])
        entity_details = state.get("preprocessor_entity_details", {})
        predicate_details = state.get("preprocessor_predicate_details", {})
        entities_by_community = state.get("preprocessor_entities_by_community", {})
        predicates_by_community = state.get("preprocessor_predicates_by_community", {})
        query_type = state.get("preprocessor_query_type", "general")
        user_query = state.get("user_query", "")

        community_plans = self._plan_per_community(
            community_ids, entities_by_community, predicates_by_community,
            entity_ids, predicate_ids,
        )
        for cp in community_plans:
            label = cp["community_id"] or "(all/ungrouped)"
            print(f"  🌳 Branch selection for {label}: {cp['branches']} "
                  f"({len(cp['entity_ids'])} entities, {len(cp['predicate_ids'])} predicates)")

        # Flat list of every (community_id, branch) pair the LLM must
        # produce exactly one spec for — used by validation/backfill below.
        required_specs = [
            (cp["community_id"], branch)
            for cp in community_plans
            for branch in cp["branches"]
        ]

        context_block = self._build_context_block(
            entity_ids, entity_details,
            predicate_ids, predicate_details,
            community_ids, query_type, community_plans,
        )

        messages = [
            SystemMessage(content=ORCHESTRATOR_SPEC_PROMPT),
            HumanMessage(
                content=f"{context_block}\n\nUser Question: {user_query}\n\n"
                        f"Emit exactly one worker_spec per (community, branch) pair listed "
                        f"above, tuning parameters per the rules in your instructions. "
                        f"Return ONLY valid JSON."
            ),
        ]

        print("\n[Orchestrator Debug] ── Spec-planning mode (single-shot) ──────")
        response = self.llm.invoke(messages)
        output = response.content
        preview = output[:300].replace("\n", " ")
        print(f"  [ai_response] {preview}{'...' if len(output) > 300 else ''}")
        print("[Orchestrator Debug] ────────────────────────────────────────────\n")

        plan = self._parse_json_from_output(output)

        if not plan or not plan.get("worker_specs"):
            print("[Orchestrator Debug] LLM JSON parse failed or empty — building specs directly")
            plan = {
                "query_type": query_type,
                "entities_found": entity_ids,
                "predicates_found": predicate_ids,
                "communities_identified": community_ids,
                "worker_specs": self._build_specs_from_community_plans(community_plans),
                "resolution_method": "preprocessor_direct",
            }
        else:
            plan = self._validate_specs(plan, required_specs, community_plans)
            plan["resolution_method"] = "preprocessor_llm"

        # _validate_specs / _build_specs_from_community_plans already
        # enforce exactly one spec per (community, branch) pair, so no
        # further truncation is needed.
        plan["strategies"] = plan["worker_specs"]  # back-compat alias
        plan["raw_output"] = output
        plan["user_query"] = user_query
        return plan

    def _build_context_block(
        self,
        entity_ids: List[str],
        entity_details: Dict[str, Any],
        predicate_ids: List[str],
        predicate_details: Dict[str, Any],
        community_ids: List[str],
        query_type: str,
        community_plans: List[Dict[str, Any]],
    ) -> str:
        """Render the preprocessor's resolved signals as a prompt context
        block, grouped per community so the LLM tunes each community's
        specs against ONLY that community's entities/predicates — never a
        flat combined list that would blur which entity belongs to which
        village in a comparison query.
        """
        import json

        def _fmt_ids(ids: List[str], details: Dict[str, Any]) -> str:
            lines = [
                f"  - {cid} (name: \"{details.get(cid, {}).get('name', '?')}\", "
                f"score: {details.get(cid, {}).get('score', 0):.2f})"
                for cid in ids
            ]
            return chr(10).join(lines) if lines else "  (none)"

        sections = []
        for cp in community_plans:
            label = cp["community_id"] or "(ungrouped — single/no community)"
            sections.append(f"""### Community: {label}
Entity Canonical IDs:
{_fmt_ids(cp["entity_ids"], entity_details)}
Predicate Canonical IDs:
{_fmt_ids(cp["predicate_ids"], predicate_details)}
Branches already selected for THIS community (emit exactly one worker_spec per branch, scoped to ONLY this community's IDs above):
{json.dumps(cp["branches"], ensure_ascii=False)}""")

        community_sections = "\n\n".join(sections)

        return f"""## Pre-Processed Context

### Query Type: {query_type}
### All Community IDs: {json.dumps(community_ids, ensure_ascii=False)}

{community_sections}

### IMPORTANT — Multi-Community Scoping
When more than one community section appears above, each is a SEPARATE worker_spec
target. A spec built for one community's branch must use ONLY that community's entity_ids/
predicate_ids and ONLY that community's community_id in its own "community_ids" field —
never mix entities from one community into a spec scoped to another, and never combine two
communities into a single spec even if they share the same branch name (e.g. both having
graph_hop). This is what keeps a comparison query from letting one community's results
crowd out another's."""

    def _validate_specs(
        self,
        plan: Dict[str, Any],
        required_specs: List[Tuple[Optional[str], str]],
        community_plans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Keep only specs whose tool is in VALID_TOOLS AND whose
        (community_id, tool) pair is in the already-decided required_specs
        list — never trust free-form model output to dictate what code
        path executes, what community a spec is scoped to, or to invent a
        (community, branch) pair Python didn't already select.

        A spec's community_id is read from its "community_ids" field
        (taking the first/only entry — each required spec is scoped to
        exactly one community, or None for the single/no-community case).
        If the model dropped a required (community, branch) pair or
        duplicated one, backfill/dedupe deterministically so the final
        spec list always has exactly one entry per pair in required_specs.
        """
        specs = plan.get("worker_specs", [])
        by_pair: Dict[Tuple[Optional[str], str], Dict[str, Any]] = {}

        for spec in specs:
            tool = spec.get("tool")
            if tool not in VALID_TOOLS:
                print(f"[Orchestrator Debug] Dropping spec with invalid tool name: {tool!r}")
                continue

            spec_community_ids = spec.get("community_ids") or []
            spec_cid = spec_community_ids[0] if spec_community_ids else None

            pair = (spec_cid, tool)
            if pair not in required_specs:
                print(f"[Orchestrator Debug] Dropping spec for {pair!r} — not in required (community, branch) pairs")
                continue
            if pair in by_pair:
                print(f"[Orchestrator Debug] Duplicate spec for {pair!r} — keeping first, dropping extra")
                continue

            spec.setdefault("parameters", {})
            spec["parameters"].setdefault("predicate_properties", [])
            by_pair[pair] = spec

        missing = [p for p in required_specs if p not in by_pair]
        if missing:
            print(f"[Orchestrator Debug] LLM omitted required (community, branch) pair(s) {missing} — backfilling deterministically")
            backfilled = self._build_specs_from_community_plans(
                community_plans, only_pairs=set(missing)
            )
            for spec in backfilled:
                spec_cid = (spec.get("community_ids") or [None])[0]
                by_pair[(spec_cid, spec["tool"])] = spec

        # Preserve original required_specs order, not dict insertion order.
        plan["worker_specs"] = [by_pair[p] for p in required_specs if p in by_pair]
        return plan

    def _build_specs_from_community_plans(
        self,
        community_plans: List[Dict[str, Any]],
        only_pairs: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Deterministic fallback/backfill: build a default-parameter spec
        for each (community, branch) pair across all community_plans. Used
        both when the LLM's JSON can't be parsed at all, and to backfill
        any (community, branch) pair the LLM's output omitted.

        Args:
            community_plans: output of _plan_per_community
            only_pairs: if given, restrict to these (community_id, tool)
                pairs only (used for backfill — don't rebuild specs that
                already came from the LLM successfully)

        predicate_properties defaults to empty here since there's no LLM
        judgment available to suggest property names in this path — the
        worker still returns whatever properties exist on each edge
        regardless (see markdown_tools._slim_rel_properties), only the
        "which properties matter for this question" hint is lost.
        """
        specs: List[Dict[str, Any]] = []

        for cp in community_plans:
            cid = cp["community_id"]
            entity_ids = cp["entity_ids"]
            predicate_ids = cp["predicate_ids"]
            branches = cp["branches"]
            community_ids_field = [cid] if cid else []

            if "graph_hop" in branches and (only_pairs is None or (cid, "graph_hop") in only_pairs):
                specs.append({
                    "tool": "graph_hop",
                    "name": f"entity_neighborhood_{cid or 'all'}",
                    "entity_ids": entity_ids,
                    "predicate_ids": predicate_ids,
                    "chunk_ids": [],
                    "community_ids": community_ids_field,
                    "parameters": {"depth": 1, "limit": 50, "predicate_properties": []},
                })

            if "find_paths" in branches and (only_pairs is None or (cid, "find_paths") in only_pairs):
                specs.append({
                    "tool": "find_paths",
                    "name": f"entity_path_search_{cid or 'all'}",
                    "entity_ids": entity_ids,
                    "predicate_ids": [],
                    "chunk_ids": [],
                    "community_ids": community_ids_field,
                    "parameters": {"depth": 3, "limit": 20, "predicate_properties": []},
                })

            if "community_explore" in branches and (only_pairs is None or (cid, "community_explore") in only_pairs):
                specs.append({
                    "tool": "community_explore",
                    "name": f"community_relationship_scan_{cid or 'all'}",
                    "entity_ids": [],
                    "predicate_ids": [],
                    "chunk_ids": [],
                    "community_ids": community_ids_field,
                    "parameters": {"limit": 100, "predicate_properties": []},
                })

        return specs

    def _parse_json_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Try multiple methods to extract JSON from LLM output."""
        import json
        import re

        match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match = re.search(r'```\s*(\{.*?\})\s*```', output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        start_idx = output.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(output)):
                if output[i] == '{':
                    brace_count += 1
                elif output[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            if end_idx > start_idx:
                try:
                    return json.loads(output[start_idx:end_idx])
                except json.JSONDecodeError:
                    pass

        return None

    # ------------------------------------------------------------------
    # ORIGINAL ReAct-based fallback path — used only when the preprocessor
    # itself failed (state has no preprocessor_entity_ids/predicate_ids/
    # chunk_ids at all) and the orchestrator must resolve canonical IDs
    # itself via tool calls. Unchanged in spirit from the original; still
    # emits legacy "approach"-style strategies, but worker.py's TOOL_MAP
    # maps "direct"/"expansion"/"path"/"community_exploration" approach
    # strings onto the same dispatch table as the new tool names, so this
    # fallback path still executes correctly without modification.
    # ------------------------------------------------------------------

    def analyze_and_plan(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and plan query strategies via ReAct tool calls.

        Used only when the preprocessor pipeline failed entirely and no
        canonical IDs/chunk IDs were resolved ahead of time.

        Args:
            user_query: The user's natural language question

        Returns:
            Dictionary containing strategies and metadata
        """
        enhanced_query = f"""{ORCHESTRATOR_SYSTEM_PROMPT}

User Question: {user_query}

Please:
1. **Detect if this is a community query**: Check if the question mentions a specific
   village or moo/หมู่. If so, call `get_community_ids_from_relationships()` to see
   available communities, then match the query to the correct community_id and include
   it in every strategy's `community_ids` list.

2. **Find canonical entities/predicates**: Use get_canonical_entities and
   get_canonical_predicates to resolve the question's key terms to canonical IDs.

3. **Apply the strict strategy rule** from your instructions above based on how many
   canonical entity IDs you found — 0 entities+predicates -> community_exploration only,
   1 entity -> expansion only, 2+ entities -> expansion AND path. Do not add any other
   strategy shape.

**IMPORTANT**: Return ONLY valid JSON with actual data from tool results, not template
placeholders.

Output your final plan in this JSON format:
{{
  "query_type": "community" | "general",
  "entities_found": ["uuid-1", "uuid-2"],
  "predicates_found": ["uuid-3"],
  "communities_identified": ["community_id_1"],
  "strategies": [
    {{
      "name": "entity_expansion",
      "description": "Expand 1-hop neighborhood relationships from the entities",
      "approach": "expansion",
      "canonical_ids": {{
        "entities": ["uuid-1", "uuid-2"],
        "predicates": []
      }},
      "community_ids": ["community_id_1"],
      "parameters": {{"depth": 1, "limit": 50, "predicate_properties": []}}
    }}
  ]
}}

Replace the example UUIDs above with actual canonical_ids from your tool calls. Do NOT use placeholder text like "[...list...]" - use real values. The "approach" field must be exactly one of "expansion", "path", "community_exploration" — never "direct" or anything else."""

        messages = [HumanMessage(content=enhanced_query)]
        result = self.agent.invoke({"messages": messages})

        import json
        import re
        from langchain_core.messages import AIMessage, ToolMessage

        output = ""
        intermediate_steps = []
        pending_tool_calls: dict = {}

        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content:
                    output += msg.content + "\n"

                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            pending_tool_calls[tc["id"]] = tc

                elif isinstance(msg, ToolMessage):
                    tc = pending_tool_calls.pop(msg.tool_call_id, None)
                    if tc:
                        intermediate_steps.append((tc, msg.content))

        plan = self._parse_json_from_output(output)
        if plan:
            plan["resolution_method"] = "matched"

        if not plan:
            plan = self._create_basic_plan_from_steps(intermediate_steps)
            if not plan:
                plan = self._try_label_group_expansion(user_query)

        if not plan or "strategies" not in plan or not plan["strategies"]:
            label_expansion_plan = self._try_label_group_expansion(user_query)
            if label_expansion_plan and label_expansion_plan.get("strategies"):
                plan = label_expansion_plan
                plan["resolution_method"] = "label_group_expansion"
            else:
                fallback = self._create_connectivity_fallback_plan()
                if plan:
                    plan["strategies"] = fallback["strategies"]
                    plan.setdefault("resolution_method", "fallback_top_connected")
                else:
                    plan = fallback

        plan["strategies"] = self._validate_react_strategies(plan.get("strategies", []))
        plan["worker_specs"] = plan["strategies"]  # forward-compat alias
        plan["raw_output"] = output
        plan["user_query"] = user_query

        return plan

    @staticmethod
    def _validate_react_strategies(strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop any ReAct-path strategy whose approach isn't one of the
        three allowed values — same never-trust-free-form-output principle
        as _validate_specs on the primary path. The ReAct agent's own JSON
        output has no structural enforcement otherwise, since it comes
        from a tool-calling loop rather than a single constrained call.
        """
        allowed_approaches = {"expansion", "path", "community_exploration"}
        valid = []
        for strat in strategies:
            approach = strat.get("approach")
            if approach in allowed_approaches:
                strat.setdefault("parameters", {})
                strat["parameters"].setdefault("predicate_properties", [])
                valid.append(strat)
            else:
                print(f"[Orchestrator Debug] Dropping ReAct strategy with invalid approach: {approach!r}")
        return valid

    def _build_comprehensive_strategies(
        self,
        entity_ids: List[str],
        predicate_ids: List[str],
        community_ids: List[str],
        strategy_prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """Build query strategies for the ReAct fallback path — restricted
        to the same strict trichotomy as the primary preprocessor-fed path
        (see Orchestrator._select_branches):
          - 0 entities AND 0 predicates -> community_exploration only
          - 1 entity                    -> expansion only
          - 2+ entities                 -> expansion AND path

        Emits legacy "approach" keys ("expansion"/"path"/"community_exploration"),
        which worker.py's TOOL_MAP still understands for this fallback path.
        No "direct" strategy, no predicate-only strategy, no deep_expansion —
        those approaches are no longer part of the orchestrator's vocabulary
        in either path.
        """
        strategies: List[Dict[str, Any]] = []

        if not entity_ids and not predicate_ids:
            strategies.append({
                "name": f"{strategy_prefix}community_exploration",
                "description": "Explore all relationships filtered by community_id — "
                                "no entity/predicate anchor was found",
                "approach": "community_exploration",
                "canonical_ids": {"entities": [], "predicates": []},
                "community_ids": community_ids,
                "parameters": {"limit": 100, "predicate_properties": []},
            })
            return strategies

        if entity_ids:
            strategies.append({
                "name": f"{strategy_prefix}entity_expansion",
                "description": "Expand 1-hop neighborhood relationships from the entities",
                "approach": "expansion",
                "canonical_ids": {"entities": entity_ids, "predicates": predicate_ids},
                "community_ids": community_ids,
                "parameters": {"depth": 1, "limit": 50, "predicate_properties": []},
            })

        if entity_ids and len(entity_ids) >= 2:
            strategies.append({
                "name": f"{strategy_prefix}entity_paths",
                "description": "Find paths connecting the identified entities",
                "approach": "path",
                "canonical_ids": {"entities": entity_ids, "predicates": []},
                "community_ids": community_ids,
                "parameters": {"depth": 3, "limit": 20, "predicate_properties": []},
            })

        return strategies

    def _try_label_group_expansion(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Try to find entities via label group expansion. Unchanged from
        the original — used only inside the ReAct fallback path."""
        import json
        import re
        import yaml
        from pathlib import Path

        try:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "label_group_config.yaml"

            with open(config_path) as f:
                label_groups = yaml.safe_load(f)

            label_group_list = "\n".join([f"- {key}: {value}" for key, value in label_groups.items()])

            identification_prompt = f"""Based on the user's question and the available label groups, identify the most relevant label group(s).

User Question: {user_query}

Available Label Groups:
{label_group_list}

Respond with ONLY a JSON object in this format:
{{
  "relevant_groups": ["GroupName1", "GroupName2"],
  "reasoning": "Brief explanation of why these groups are relevant"
}}

If multiple groups are relevant, list up to 3 in order of relevance. Use exact group names from the list above."""

            messages = [HumanMessage(content=identification_prompt)]
            response = self.llm.invoke(messages)

            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response.content, re.DOTALL)

            if not json_match:
                return None

            group_data = json.loads(json_match.group(1))
            relevant_groups = group_data.get("relevant_groups", [])

            if not relevant_groups:
                return None

            all_labels = []
            for group in relevant_groups[:3]:
                try:
                    from kg_reasoning.agents.tools.qdrant_tools import _get_manager
                    manager = _get_manager()
                    labels = manager.get_labels_by_group(group, limit=50)
                    all_labels.extend(labels)
                except Exception:
                    pass

            if not all_labels:
                return None

            labels_summary = "\n".join([
                f"- {label['name']} (group: {label.get('group', 'N/A')}, type: {label.get('type', 'N/A')})"
                for label in all_labels[:100]
            ])

            selection_prompt = f"""Based on the user's question, select the most relevant labels from the list below.

User Question: {user_query}

Available Labels:
{labels_summary}

Respond with ONLY a JSON object in this format:
{{
  "selected_labels": ["LabelName1", "LabelName2", "LabelName3"],
  "reasoning": "Brief explanation of why these labels are most relevant"
}}

Select up to 5 most relevant labels. Use exact label names from the list above."""

            messages = [HumanMessage(content=selection_prompt)]
            response = self.llm.invoke(messages)

            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response.content, re.DOTALL)

            if not json_match:
                return None

            selection_data = json.loads(json_match.group(1))
            selected_labels = selection_data.get("selected_labels", [])

            if not selected_labels:
                return None

            from kg_reasoning.agents.tools.qdrant_tools import _get_manager
            manager = _get_manager()
            entities = manager.get_entities_by_labels(selected_labels, limit=50)

            if not entities:
                return None

            entity_ids = [e["canonical_id"] for e in entities if e.get("canonical_id")]

            if not entity_ids:
                return None

            strategies = self._build_comprehensive_strategies(
                entity_ids=entity_ids,
                predicate_ids=[],
                community_ids=[],
                strategy_prefix="label_based_",
            )

            return {
                "entities_found": entity_ids,
                "predicates_found": [],
                "communities_identified": [],
                "label_groups_used": relevant_groups,
                "labels_selected": selected_labels,
                "resolution_method": "label_group_expansion",
                "strategies": strategies,
            }

        except Exception as e:
            print(f"[Label Expansion] Error during label group expansion: {e}")
            return None

    def _create_connectivity_fallback_plan(self) -> Dict[str, Any]:
        """Build a fallback plan seeded by the top 20 most connected nodes."""
        from kg_reasoning.agents.tools.neo4j_tools import _get_manager as _get_neo4j_manager

        try:
            manager = _get_neo4j_manager()
            cypher = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC
            LIMIT 20
            RETURN n.canonical_id AS canonical_id, n.name AS name, degree
            """
            rows = manager.execute_cypher_query(cypher, {}, 20)
            top_ids = [r["canonical_id"] for r in rows if r.get("canonical_id")]
        except Exception:
            top_ids = []

        strategies = self._build_comprehensive_strategies(
            entity_ids=top_ids,
            predicate_ids=[],
            community_ids=[],
            strategy_prefix="top_connected_",
        )

        return {
            "entities_found": top_ids,
            "predicates_found": [],
            "communities_identified": [],
            "resolution_method": "fallback_top_connected",
            "strategies": strategies,
        }

    def _create_basic_plan_from_steps(
        self, intermediate_steps: List[tuple]
    ) -> Optional[Dict[str, Any]]:
        """Create a basic plan from intermediate tool outputs."""
        import json

        entities = []
        predicates = []
        communities = []

        for action, observation in intermediate_steps:
            tool_name = ""
            if hasattr(action, 'tool'):
                tool_name = action.tool
            elif hasattr(action, 'get'):
                tool_name = action.get('name', '')
            elif isinstance(action, dict):
                tool_name = action.get('name', '')

            try:
                obs_data = json.loads(observation) if isinstance(observation, str) else observation

                if "get_canonical_entities" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, dict) and "canonical_id" in item:
                                entities.append(item["canonical_id"])

                elif "get_canonical_predicates" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, dict) and "canonical_id" in item:
                                predicates.append(item["canonical_id"])

                elif "get_community_metadata" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, dict) and "community_id" in item:
                                communities.append(item["community_id"])

                elif "get_community_ids_from_relationships" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, str):
                                communities.append(item)

            except Exception:
                pass

        entity_list = list(set(entities))
        predicate_list = list(set(predicates))
        community_list = list(set(communities))

        if not entity_list and not predicate_list:
            return None

        strategies = self._build_comprehensive_strategies(
            entity_ids=entity_list,
            predicate_ids=predicate_list,
            community_ids=community_list,
            strategy_prefix="partial_match_",
        )

        return {
            "entities_found": entity_list,
            "predicates_found": predicate_list,
            "communities_identified": community_list,
            "resolution_method": "partial_match",
            "strategies": strategies,
        }