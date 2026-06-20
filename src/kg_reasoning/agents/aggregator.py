"""Aggregator for multi-agent reasoning system.

Runs after workers, before the synthesizer. Reads every markdown file the
workers wrote, extracts their individual JSON result blocks, deduplicates
identical triples/paths that multiple worker specs surfaced independently,
scores each unique result against the user's query, and writes ONE file —
grouped by community_id, exactly per spec:

```markdown
# {community_id}
## relationships
{subject} -> ({predicate}) -> {object} | {attribute} | {causal link} | {temporal link}
{raw name} -> (linked) -> {canonical name}
## relevant content (references for relationships)
{relevant chunk text}
```

Reranking (the scoring engine below) is unchanged from before — only what
happens AFTER reranking changed: instead of one flat list of JSON blocks,
results are grouped by community_id, each relationship line surfaces
whatever extra edge properties survived markdown_tools' slimming step
(attribute / causal link / temporal link — whichever actually exist on that
edge), and two NEW inputs are threaded in from the preprocessor that the
old flat-JSON format never carried:
  - entity_details / predicate_details: canonical_id -> {name, raw_names}
    dicts, used to render the raw -> (linked) -> canonical name lines
  - chunk_texts: chunk_id -> {community_id, text} dicts (the preprocessor's
    own S3 fetch, see preprocessor.py Step 9), used to populate "relevant
    content" with real source text instead of nothing

SCORING APPROACH (v2 — semantic signal-aware, unchanged):
Naive keyword overlap between the query and result text fails for knowledge
graph triples because:
  - The query says "issues" but the graph stores "OralHealthProblem",
    "EnvironmentalImpact", "IMPACT_ON" — none of which share tokens with
    "issues" or "problems"
  - Community ID tokens ("village", "1", "หมู่") match every result equally
    since all results share the same community_id, making the overlap score
    a constant that doesn't distinguish anything

The v2 scorer combines three signals:
1. Query token overlap (same as before, but stripped of community/location
   tokens that appear in every result and add noise)
2. Semantic type signal — node types and predicate names are mapped to
   a relevance weight based on the query's intent category (issues/problems,
   people, activities, organizations, etc.)
3. Predicate directionality bonus — predicates like IMPACT_ON,
   HAS_AFFECTING_FACTORS, CREATES_IMPACT_FROM score higher for issue queries
   than structural predicates like IS, HAS_EXPERTISE_IN, HOLDS_EVENTS_OF
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from kg_reasoning.agents.tools.markdown_tools import _get_manager as _get_md_manager


_RESULT_BLOCK_RE = re.compile(
    r"### Result \d+\s*\n```json\s*\n(.*?)\n```", re.DOTALL
)

_WORD_RE = re.compile(r"[a-zA-Z0-9ก-๙]+", re.UNICODE)

# Query intent signals → which node types and predicates carry weight
_ISSUE_QUERY_TOKENS: Set[str] = {
    "issue", "issues", "problem", "problems", "challenge", "challenges",
    "concern", "concerns", "difficulty", "difficulties", "obstacle", "obstacles",
    "ปัญหา", "อุปสรรค", "ความยาก",
}

_PEOPLE_QUERY_TOKENS: Set[str] = {
    "who", "person", "people", "leader", "leaders", "member", "members",
    "resident", "residents", "คน", "ผู้นำ", "สมาชิก",
}

_ACTIVITY_QUERY_TOKENS: Set[str] = {
    "activity", "activities", "event", "events", "do", "does", "doing",
    "กิจกรรม", "งาน",
}

# Node type → base relevance score for each intent category
# Shape: {node_type_lowercase: {intent: score}}
_TYPE_SCORES: Dict[str, Dict[str, float]] = {
    "impact":                   {"issue": 1.0, "people": 0.1, "activity": 0.2},
    "directimpact":             {"issue": 1.0, "people": 0.1, "activity": 0.2},
    "healthimpact":             {"issue": 1.0, "people": 0.3, "activity": 0.1},
    "environmentalimpact":      {"issue": 1.0, "people": 0.1, "activity": 0.2},
    "environmentalimpactvalue": {"issue": 0.9, "people": 0.1, "activity": 0.1},
    "socialimpact":             {"issue": 0.9, "people": 0.3, "activity": 0.2},
    "economicimpact":           {"issue": 0.9, "people": 0.1, "activity": 0.1},
    "oralhealthproblem":        {"issue": 1.0, "people": 0.2, "activity": 0.1},
    "researchtopic":            {"issue": 0.7, "people": 0.1, "activity": 0.2},
    "person":                   {"issue": 0.2, "people": 1.0, "activity": 0.3},
    "leader":                   {"issue": 0.2, "people": 1.0, "activity": 0.3},
    "communityworkergroup":     {"issue": 0.3, "people": 0.6, "activity": 0.5},
    "waterusergroup":           {"issue": 0.3, "people": 0.4, "activity": 0.4},
    "socialcapitalactivity":    {"issue": 0.2, "people": 0.3, "activity": 0.8},
    "waterupplyservice":        {"issue": 0.3, "people": 0.1, "activity": 0.5},
}

# Predicate → base relevance score for each intent
_PREDICATE_SCORES: Dict[str, Dict[str, float]] = {
    "IMPACT_ON":                    {"issue": 1.0, "people": 0.2, "activity": 0.2},
    "HAS_AFFECTING_FACTORS":        {"issue": 0.9, "people": 0.1, "activity": 0.2},
    "CREATES_IMPACT_FROM":          {"issue": 0.9, "people": 0.1, "activity": 0.2},
    "IMPACTS_REDUCTION_OF":         {"issue": 0.8, "people": 0.1, "activity": 0.1},
    "HAS_PROBLEM":                  {"issue": 1.0, "people": 0.1, "activity": 0.1},
    "CAUSES":                       {"issue": 0.8, "people": 0.1, "activity": 0.1},
    "LEADS_TO":                     {"issue": 0.7, "people": 0.2, "activity": 0.3},
    "IS_MEMBER_OF":                 {"issue": 0.1, "people": 0.9, "activity": 0.3},
    "IS_LEADER_OF":                 {"issue": 0.1, "people": 1.0, "activity": 0.3},
    "HAS_EXPERTISE_IN":             {"issue": 0.1, "people": 0.2, "activity": 0.5},
    "IS":                           {"issue": 0.0, "people": 0.1, "activity": 0.1},
    "HOLDS_EVENTS_AND_ACTIVITIES_OF_SOCIAL_CAPITAL": {"issue": 0.1, "people": 0.2, "activity": 0.8},
    "HAS_MANAGEMENT_OF_WORK_AND_ACTIVITIES":         {"issue": 0.1, "people": 0.2, "activity": 0.7},
}

# Community/location tokens that appear in every result's community_id and
# add noise to overlap scoring — strip these before computing token overlap.
_NOISE_TOKENS: Set[str] = {
    "village", "หมู่", "หมู่บ้าน", "moo", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "10", "community", "ชุมชน",
}


def _unwrap_path(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the inner {"chain": [...], "community_ids": [...]} dict for a
    path result, or None if `result` isn't a path.

    markdown_tools._slim_result's path branch returns
    {"p": {"chain": [...], "community_ids": [...]}} or {"path": {...}} —
    it preserves whatever Cypher RETURN variable name was used (worker.py's
    find_paths query does `RETURN p`), wrapping the chain dict one level
    deeper than every other helper in this file assumed. Every place that
    needs to read a path's chain/community_ids must unwrap through this
    function first — checking `"chain" in result` directly on the outer
    dict will never match, since the outer dict's only key is "p"/"path".
    """
    if not isinstance(result, dict):
        return None
    for key in ("p", "path"):
        inner = result.get(key)
        if isinstance(inner, dict) and "chain" in inner:
            return inner
    # Fallback: already-unwrapped shape (e.g. constructed directly in a
    # test, or a future caller that already unwrapped it) — accept it too.
    if "chain" in result:
        return result
    return None


def _tokenize(text: str) -> Set[str]:
    """Lowercase word-set tokenizer including Thai Unicode block."""
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _detect_intent(query_tokens: Set[str]) -> str:
    """Detect the dominant intent category from the query token set."""
    issue_overlap = len(query_tokens & _ISSUE_QUERY_TOKENS)
    people_overlap = len(query_tokens & _PEOPLE_QUERY_TOKENS)
    activity_overlap = len(query_tokens & _ACTIVITY_QUERY_TOKENS)

    scores = {"issue": issue_overlap, "people": people_overlap, "activity": activity_overlap}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


def _score_result(result: Dict[str, Any], query_tokens: Set[str], intent: str) -> float:
    """Score a single result using token overlap + semantic type/predicate signals.

    Returns a float in [0, 2.0] — higher is more relevant.
    """
    # Strip noise tokens before overlap — community_id tokens appear in
    # every result and drown out meaningful signal.
    clean_query = query_tokens - _NOISE_TOKENS

    # Collect text tokens from result, also stripping noise
    flat_text = _flatten_text(result)
    result_tokens = _tokenize(flat_text) - _NOISE_TOKENS

    token_overlap = (
        len(clean_query & result_tokens) / max(len(clean_query), 1)
        if clean_query else 0.0
    )

    # Semantic type signal
    subject_type = ""
    object_type = ""
    predicate = ""

    if "subject" in result and isinstance(result["subject"], dict):
        subject_type = (result["subject"].get("type") or "").lower()
    if "object" in result and isinstance(result["object"], dict):
        object_type = (result["object"].get("type") or "").lower()
    if "predicate" in result:
        predicate = result.get("predicate", "")

    # For evidence chunk results
    if "evidence_quote" in result or "evidence_quote_en" in result:
        quote_tokens = _tokenize(
            result.get("evidence_quote_en", "") + " " + result.get("evidence_quote", "")
        ) - _NOISE_TOKENS
        quote_overlap = len(clean_query & quote_tokens) / max(len(clean_query), 1) if clean_query else 0.0
        return min(2.0, token_overlap + quote_overlap)

    # For path/chain results
    path_data = _unwrap_path(result)
    if path_data is not None:
        chain = path_data.get("chain", [])
        chain_text = " ".join(
            str(n.get("name", "")) if isinstance(n, dict) else str(n)
            for n in chain
        )
        chain_tokens = _tokenize(chain_text) - _NOISE_TOKENS
        chain_overlap = len(clean_query & chain_tokens) / max(len(clean_query), 1) if clean_query else 0.0
        return min(2.0, token_overlap + chain_overlap)

    # Type-based semantic score
    type_score = 0.0
    if intent != "general":
        subj_weight = _TYPE_SCORES.get(subject_type, {}).get(intent, 0.0)
        obj_weight = _TYPE_SCORES.get(object_type, {}).get(intent, 0.0)
        type_score = max(subj_weight, obj_weight)

    # Predicate-based semantic score
    pred_score = _PREDICATE_SCORES.get(predicate, {}).get(intent, 0.0) if intent != "general" else 0.0

    # Also check if object/subject name contains query-relevant text
    name_tokens = (
        _tokenize(result.get("subject", {}).get("name", "") if isinstance(result.get("subject"), dict) else "")
        | _tokenize(result.get("object", {}).get("name", "") if isinstance(result.get("object"), dict) else "")
    ) - _NOISE_TOKENS
    name_overlap = len(clean_query & name_tokens) / max(len(clean_query), 1) if clean_query else 0.0

    return min(2.0, token_overlap + type_score + pred_score + name_overlap)


def _flatten_text(result: Dict[str, Any]) -> str:
    """Pull every string value out of a slimmed result dict."""
    parts: List[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list):
            for v in value:
                _walk(v)

    _walk(result)
    return " ".join(parts)


def _result_key(result: Dict[str, Any]) -> Tuple:
    """Build a dedup key for a slimmed result."""
    if "subject" in result and "predicate" in result and "object" in result:
        subj_id = (result.get("subject") or {}).get("canonical_id", "")
        obj_id = (result.get("object") or {}).get("canonical_id", "")
        return ("triple", subj_id, result.get("predicate", ""), obj_id)

    if "chunk_id" in result:
        return ("chunk", result["chunk_id"])

    path_data = _unwrap_path(result)
    if path_data is not None:
        chain_ids = tuple(
            n.get("canonical_id", "") for n in path_data.get("chain", []) if isinstance(n, dict)
        )
        return ("path", chain_ids)

    return ("other", json.dumps(result, sort_keys=True, ensure_ascii=False))


def _extract_results_from_file(filepath: str) -> List[Dict[str, Any]]:
    """Parse the ```json blocks out of one worker markdown file."""
    path = Path(filepath)
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []

    results = []
    for match in _RESULT_BLOCK_RE.finditer(content):
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return results


def _result_community_ids(result: Dict[str, Any]) -> List[str]:
    """Every community_id a result belongs to.

    Triples (from graph_hop/community_explore) carry a single community_id
    field. Paths (from find_paths) carry a community_ids LIST inside their
    chain dict, since a path can cross more than one community — such a
    path is listed under every community it touches, not just the first.
    """
    if "community_id" in result and result["community_id"]:
        return [result["community_id"]]

    path_data = _unwrap_path(result)
    if path_data is not None:
        return [cid for cid in path_data.get("community_ids", []) if cid]

    return []


def _render_relationship_line(result: Dict[str, Any]) -> Optional[str]:
    """Render one triple as: subject -> (predicate) -> object | attr | causal | temporal

    Extra relationship properties (attribute, causal link/weight, temporal
    link — whatever markdown_tools._slim_rel_properties actually found on
    that edge) are appended after the triple, pipe-separated, in whatever
    order their property keys appeared — there is no fixed schema for which
    properties exist, since edges carry them inconsistently (see
    markdown_tools._slim_rel_properties).

    Returns None for non-triple results (paths) — those are rendered
    separately by _render_path_line.
    """
    if not ("subject" in result and "predicate" in result and "object" in result):
        return None

    subj = result.get("subject") or {}
    obj = result.get("object") or {}
    subj_name = subj.get("name", "?") if isinstance(subj, dict) else str(subj)
    obj_name = obj.get("name", "?") if isinstance(obj, dict) else str(obj)
    predicate = result.get("predicate", "?")

    line = f"{subj_name} -> ({predicate}) -> {obj_name}"

    properties = result.get("properties") or {}
    extra = [f"{k}: {v}" for k, v in properties.items() if v is not None]
    if extra:
        line += " | " + " | ".join(extra)

    return line


def _render_path_line(result: Dict[str, Any]) -> Optional[str]:
    """Render a path chain as a single relationship-style line.

    Path chains alternate node/relationship entries (see
    markdown_tools._slim_path): [node, rel, node, rel, node, ...]. Rendered
    as subject -> (pred) -> object -> (pred) -> object ..., with any edge
    properties pipe-separated after the whole chain rather than per-hop,
    since a multi-hop path's properties are most useful as a single
    combined reference line rather than fragmented mid-sentence.
    """
    path_data = _unwrap_path(result)
    if path_data is None:
        return None

    chain = path_data.get("chain")
    if not chain:
        return None

    parts: List[str] = []
    all_properties: Dict[str, Any] = {}
    for item in chain:
        if isinstance(item, dict) and "predicate" in item:
            parts.append(f"-> ({item['predicate']}) ->")
            for k, v in (item.get("properties") or {}).items():
                if v is not None:
                    all_properties[k] = v
        elif isinstance(item, dict):
            parts.append(item.get("name", "?"))
        else:
            parts.append(str(item))

    line = " ".join(parts)
    extra = [f"{k}: {v}" for k, v in all_properties.items()]
    if extra:
        line += " | " + " | ".join(extra)
    return line


def _canonical_link_lines(
    result: Dict[str, Any],
    entity_details: Dict[str, Any],
    predicate_details: Dict[str, Any],
) -> List[str]:
    """Render 'raw name -> (linked) -> canonical name' lines for one result.

    entity_details / predicate_details come straight from the preprocessor
    (state["preprocessor_entity_details"] / state["preprocessor_predicate_details"]):
    canonical_id -> {"name": <canonical name>, "raw_names": [<raw surface
    forms>], "score": <float>}. A result's subject/object only carry
    canonical_id + canonical name (that's all Neo4j nodes store) — this
    looks up the SAME canonical_id in entity_details to recover every raw
    surface form that resolved to it, giving the synthesizer the link the
    preprocessor already established between what was actually extracted
    from text and what the graph calls it.

    One line per (raw_name, canonical_name) pair, deduplicated across a
    single result (a triple touches up to 3 canonical things: subject,
    predicate, object).
    """
    lines: List[str] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    def _add_for(canonical_id: str, details: Dict[str, Any]) -> None:
        info = details.get(canonical_id)
        if not info:
            return
        canonical_name = info.get("name", "")
        for raw_name in info.get("raw_names", []):
            if not raw_name or raw_name == canonical_name:
                # Skip when the raw form and canonical form are identical —
                # there is nothing to link, it would just be noise.
                continue
            pair = (raw_name, canonical_name)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                lines.append(f"{raw_name} -> (linked) -> {canonical_name}")

    subj = result.get("subject")
    if isinstance(subj, dict) and subj.get("canonical_id"):
        _add_for(subj["canonical_id"], entity_details)

    obj = result.get("object")
    if isinstance(obj, dict) and obj.get("canonical_id"):
        _add_for(obj["canonical_id"], entity_details)

    # Predicates are stored on triples as a canonical NAME string, not a
    # canonical_id (see markdown_tools._slim_rel_type) — predicate_details
    # is keyed by canonical_id, so match by name instead for this field.
    predicate_name = result.get("predicate")
    if predicate_name:
        for pid, info in predicate_details.items():
            if info.get("name") == predicate_name:
                for raw_name in info.get("raw_names", []):
                    if raw_name and raw_name != predicate_name:
                        pair = (raw_name, predicate_name)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            lines.append(f"{raw_name} -> (linked) -> {predicate_name}")
                break

    # Path chains: walk node entries for subject/object-style canonical links.
    path_data = _unwrap_path(result)
    if path_data is not None:
        for item in path_data.get("chain", []):
            if isinstance(item, dict) and "canonical_id" in item:
                _add_for(item["canonical_id"], entity_details)

    return lines


def split_by_community(content: str) -> List[Tuple[str, str]]:
    """Split an aggregated file's content into (community_id, section_text)
    pairs, using the COMMUNITY_START/COMMUNITY_END markers _write_aggregated_file
    emits — not markdown heading parsing, which could collide with content
    that happens to start with "# " (a chunk's source text, an unusual
    community_id, etc.).

    Used by synthesizer.py's map-reduce path (3+ communities) to summarize
    each community's section independently before a final combine step.
    Each returned section_text includes everything between (not including)
    its own START/END marker lines — the "# {cid}" heading, "## relationships",
    relationship lines, canonical links, "## relevant content", and chunk
    text for that one community.

    Returns an empty list if no markers are found (e.g. content is empty,
    or pre-dates this marker format) — callers should treat that as "cannot
    split, fall back to single-call handling" rather than an error.
    """
    pattern = re.compile(
        r"<!-- COMMUNITY_START: (.*?) -->\n(.*?)<!-- COMMUNITY_END: \1 -->",
        re.DOTALL,
    )
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(content)]


def aggregate_results(
    markdown_files: List[str],
    user_query: str,
    entity_details: Optional[Dict[str, Any]] = None,
    predicate_details: Optional[Dict[str, Any]] = None,
    chunk_texts: Optional[Dict[str, Any]] = None,
    community_labels: Optional[Dict[str, str]] = None,
    max_results: int = 200,
) -> Dict[str, Any]:
    """Deduplicate and rank results across all worker output files, then
    write ONE file grouped by community_id with relationships, raw->canonical
    links, and reference chunk text per community.

    Args:
        markdown_files: Paths written by workers (state["markdown_files"])
        user_query: Original user question, used for relevance scoring
        entity_details: state["preprocessor_entity_details"] — canonical_id ->
            {"name", "raw_names", "score"}, used for raw->canonical link lines
        predicate_details: state["preprocessor_predicate_details"] — same
            shape as entity_details, for predicates
        chunk_texts: state["preprocessor_chunk_texts"] — chunk_id ->
            {"community_id", "text", "s3_key", "error"}, the preprocessor's
            own S3 fetch (see preprocessor.py Step 9) — used to populate the
            "relevant content" section per community
        community_labels: state["preprocessor_community_labels"] —
            community_id -> human-readable label (e.g. village name), from
            metadata_registry. Rendered into each community's heading so
            the synthesizer never has to infer "Village 1" from an opaque
            internal ID string like หมู่-1_village-1 — that inference is
            exactly the kind of thing an LLM can get wrong, especially when
            sibling IDs are inconsistently formatted (a missing hyphen
            between two otherwise-parallel village IDs, for example).
        max_results: Hard cap on unique results passed to the synthesizer

    Returns:
        {"filepath": <path to aggregated .md>, "result_count": <int>}
    """
    entity_details = entity_details or {}
    predicate_details = predicate_details or {}
    chunk_texts = chunk_texts or {}
    community_labels = community_labels or {}

    query_tokens = _tokenize(user_query)
    intent = _detect_intent(query_tokens)
    print(f"  [Aggregator] Detected query intent: {intent}")

    seen: Dict[Tuple, Dict[str, Any]] = {}

    for filepath in markdown_files:
        for result in _extract_results_from_file(filepath):
            key = _result_key(result)
            score = _score_result(result, query_tokens, intent)

            existing = seen.get(key)
            if existing is None or score > existing["score"]:
                seen[key] = {"result": result, "score": score}

    if not seen:
        return {"filepath": None, "result_count": 0}

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:max_results]

    # Log top-5 scores for debugging
    print(f"  [Aggregator] Top 5 scores:")
    for item in ranked[:5]:
        r = item["result"]
        if _unwrap_path(r) is not None:
            desc = _render_path_line(r) or "[empty path]"
        else:
            subj = r.get("subject", {}).get("name", "?") if isinstance(r.get("subject"), dict) else "?"
            pred = r.get("predicate", "?")
            obj = r.get("object", {}).get("name", "?") if isinstance(r.get("object"), dict) else "?"
            desc = f"{subj} —[{pred}]→ {obj}"
        print(f"    {item['score']:.3f}  {desc}")

    filepath = _write_aggregated_file(
        ranked, user_query, entity_details, predicate_details, chunk_texts, community_labels
    )
    return {"filepath": filepath, "result_count": len(ranked)}


def _write_aggregated_file(
    ranked: List[Dict[str, Any]],
    user_query: str,
    entity_details: Dict[str, Any],
    predicate_details: Dict[str, Any],
    chunk_texts: Dict[str, Any],
    community_labels: Dict[str, str],
) -> str:
    """Write ranked, deduplicated results grouped by community_id:

    <!-- COMMUNITY_START: {community_id} -->
    # {community_id} ({label})
    ## relationships
    {subject} -> ({predicate}) -> {object} | {attribute} | {causal} | {temporal}
    {raw name} -> (linked) -> {canonical name}
    ## relevant content (references for relationships)
    {chunk text}
    <!-- COMMUNITY_END: {community_id} -->

    The heading includes community_labels[community_id] (e.g. "Village 2")
    alongside the raw internal community_id, when a label is available —
    this is what lets the synthesizer match the user's "village 1"/
    "village 2" phrasing to the right section without inferring it from an
    opaque, inconsistently-formatted internal ID string.

    Results with no resolvable community_id (shouldn't normally happen,
    since every graph_hop/find_paths/community_explore result carries one —
    see aggregator._result_community_ids) are grouped under "unknown" rather
    than silently dropped, so nothing reranked is ever lost from the output.
    A result touching multiple communities (a path crossing village
    boundaries) is listed under every community it touches.
    """
    md = _get_md_manager()

    # Group ranked results by every community_id they touch, preserving
    # rank order within each group (ranked is already score-sorted). Track
    # each community's best (lowest) rank index in the same pass so we can
    # order communities by their most relevant result without a second,
    # quadratic lookup pass.
    by_community: Dict[str, List[Dict[str, Any]]] = {}
    best_rank: Dict[str, int] = {}
    for idx, item in enumerate(ranked):
        result = item["result"]
        cids = _result_community_ids(result) or ["unknown"]
        for cid in cids:
            by_community.setdefault(cid, []).append(result)
            if cid not in best_rank:
                best_rank[cid] = idx

    # Community order: by the best (highest-ranked) result in each group, so
    # communities with more relevant content surface first — consistent
    # with the reranking the rest of this module already does.
    community_order = sorted(by_community.keys(), key=lambda cid: best_rank[cid])

    # Per-community relationship cap — `ranked` (and therefore each
    # by_community[cid] list, since it's built by iterating `ranked` in
    # order) is already score-sorted, so capping is a simple slice that
    # keeps the highest-ranked relationships for that community. Without
    # this, a single community with 100+ matches (real observed volumes)
    # could dominate the file even after the global max_results cap,
    # since that cap operates across all communities combined, not per
    # community.
    MAX_RELATIONSHIPS_PER_COMMUNITY = 40

    lines: List[str] = []
    lines.append(f"# Aggregated results for: {user_query}\n")

    for cid in community_order:
        results = by_community[cid]
        if len(results) > MAX_RELATIONSHIPS_PER_COMMUNITY:
            print(f"  ✂️  Capping relationships for {cid}: "
                  f"{len(results)} → {MAX_RELATIONSHIPS_PER_COMMUNITY}")
            results = results[:MAX_RELATIONSHIPS_PER_COMMUNITY]

        # Explicit, unambiguous boundary markers (not just the "# {cid}"
        # heading) — let callers like synthesizer.py split sections
        # reliably by string search rather than parsing markdown headings,
        # which could collide with content that happens to start with
        # "# " (a chunk's source text, an unusual community_id, etc.).
        lines.append(f"<!-- COMMUNITY_START: {cid} -->")
        label = community_labels.get(cid)
        if label and label != cid:
            # Both forms shown explicitly and unambiguously labeled, so the
            # model never has to guess which internal ID corresponds to
            # which named place the user's question used — this is the fix
            # for the comparison-query failure where one village's data was
            # present but the model couldn't connect it to "Village 1"
            # because the internal ID (หมู่-1_village-1) doesn't read as
            # "Village 1" on its own, especially next to an inconsistently
            # formatted sibling ID (หมู่-2_village2, missing the hyphen).
            lines.append(f"# {label} (internal id: {cid})\n")
        else:
            lines.append(f"# {cid}\n")
        lines.append("## relationships\n")

        canonical_link_lines: List[str] = []
        seen_canonical_pairs: Set[str] = set()

        for result in results:
            rel_line = _render_relationship_line(result)
            if rel_line is None:
                rel_line = _render_path_line(result)
            if rel_line:
                lines.append(rel_line)

            for link_line in _canonical_link_lines(result, entity_details, predicate_details):
                if link_line not in seen_canonical_pairs:
                    seen_canonical_pairs.add(link_line)
                    canonical_link_lines.append(link_line)

        if canonical_link_lines:
            lines.append("")  # blank line separating triples from link lines
            lines.extend(canonical_link_lines)

        lines.append("")
        lines.append("## relevant content (references for relationships)\n")

        community_chunks = [
            info for info in chunk_texts.values()
            if info.get("community_id") == cid and info.get("text")
        ]
        if community_chunks:
            # Backstop char cap per community — the preprocessor already
            # caps chunk COUNT per community (see
            # preprocessor._cap_chunks_per_community), but a handful of
            # unusually long chunks could still blow past a reasonable
            # context budget. Add chunks in order until the cap is hit;
            # truncate only the chunk that crosses the boundary, so
            # earlier chunks for this community are never cut mid-sentence
            # just because a later chunk happened to be long.
            char_budget = 8000
            used_chars = 0
            for idx, info in enumerate(community_chunks):
                if used_chars >= char_budget:
                    remaining_count = len(community_chunks) - idx
                    lines.append(f"*[{remaining_count} more chunk(s) omitted — "
                                  f"community content cap reached]*")
                    break
                text = info["text"]
                remaining_budget = char_budget - used_chars
                if len(text) > remaining_budget:
                    text = text[:remaining_budget] + "\n\n[truncated — community content cap reached]"
                lines.append(text)
                lines.append("")
                used_chars += len(text)
        else:
            lines.append("*No reference chunk text available for this community*\n")

        lines.append(f"<!-- COMMUNITY_END: {cid} -->\n")

    content = "\n".join(lines)

    # Same timestamped-filename convention write_query_results uses, so this
    # file doesn't collide with a previous run's aggregation and sorts
    # correctly among other files in the output directory.
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = md.output_dir / f"{timestamp}_aggregated_ranked.md"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return str(filepath)