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

SCORING APPROACH (v3 — embedding-aware, no hardcoded categories):
v2 mapped node types and predicates to hand-picked relevance weights per
query "intent" (issue/people/activity), guessed in advance. That breaks for
any query, type, or predicate whose wording wasn't anticipated — e.g. a
query about "irrigation" when the hardcoded table only knows "issue",
"people", "activity", or a new node type added to the graph after the
table was written. It's a strictly maintenance-bound approach: relevance
for unseen vocabulary is exactly zero until someone edits this file.

v3 replaces all of that with two query-aware signals, neither of which
requires anticipating vocabulary in advance:

1. EMBEDDING SIMILARITY (primary). Every triple in this pipeline is
   evidenced by a specific chunk_id (markdown_tools._slim_result pulls it
   off the originating Neo4j edge's properties), and every path hop now
   carries its own chunk_id the same way (markdown_tools._slim_path was
   updated to surface it instead of stripping it). The preprocessor
   already computed a real relevance score between the user's query and
   that exact chunk's text during its own evidence search (see
   preprocessor._restructure_by_indices /
   preprocessor._cap_chunks_per_community, which store this as
   chunk_details[chunk_id]["score"]). That score is NOT always a plain
   cosine similarity — qdrant_tools.search_evidence runs hybrid dense+
   sparse retrieval with RRF fusion when a sparse vector is configured for
   evidence_registry (the common case), and RRF's raw score is a rank-
   based quantity, not a similarity. qdrant_tools.search_evidence
   normalizes this at the source (min-max rescaled within each query's own
   RRF result set) before it ever reaches chunk_details, so by the time it
   gets here it's always a comparable [0, 1] relevance value regardless of
   which retrieval path produced it — this scorer does not need to know or
   care which one fired. Looking that score up per result means the
   embedding/retrieval layer — not a hand-written table — decides how
   relevant a result's underlying evidence is to whatever the user asked,
   for triples and multi-hop paths alike, in whatever language or
   vocabulary they used.
2. LEXICAL OVERLAP (secondary). Plain token overlap between the query and
   the result's own subject/predicate/object names (or, for paths, its
   chain's node names), stripped of community/location noise tokens. This
   catches exact-term matches (a literal entity name, a specific figure)
   that a chunk-level embedding score can blur, and costs nothing extra
   since it's derived purely from the query and the result — no category
   table involved.

chunk_details must be passed in (state["preprocessor_chunk_details"]) for
signal 1 to be available; without it, scoring degrades to lexical overlap
only, which is still no worse than v2's behavior for any term outside its
hardcoded tables.
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

# Community/location tokens that appear in every result's community_id and
# add noise to lexical overlap scoring — strip these before computing
# overlap. This is NOT a relevance judgment about what the query means (it
# doesn't claim "issue" or "people" matters more than another word) — it's
# a structural fact about this pipeline's data: every triple in a community
# carries that community's id/number tokens regardless of content, so they
# can never discriminate between results and only dilute real overlap.
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


def _result_chunk_ids(result: Dict[str, Any]) -> List[str]:
    """Every chunk_id a result is evidenced by, as strings (matching the
    string keys chunk_details/chunk_texts use — see
    preprocessor._fetch_s3_chunk_texts, which stores str(chunk_id)).

    Triples carry a single top-level chunk_id (markdown_tools._slim_result
    reads it off the Neo4j relationship's own properties). Paths carry a
    chunk_ids LIST inside their unwrapped chain dict (markdown_tools.
    _slim_path now surfaces each hop's chunk_id the same way _slim_result
    already did for triples — see that function's docstring) since a
    multi-hop path can be evidenced by more than one source chunk, one per
    hop. A result with neither shape (or hops with no resolvable chunk_id
    at all) returns an empty list rather than guessing.
    """
    chunk_id = result.get("chunk_id")
    if chunk_id is not None and chunk_id != "":
        return [str(chunk_id)]

    path_data = _unwrap_path(result)
    if path_data is not None:
        return [str(cid) for cid in path_data.get("chunk_ids", []) if cid is not None and cid != ""]

    return []


def _score_result(
    result: Dict[str, Any],
    query_tokens: Set[str],
    chunk_details: Dict[str, Dict[str, Any]],
) -> float:
    """Score a single result by combining two query-aware signals — no
    hardcoded category/type/predicate tables:

    1. EMBEDDING SIMILARITY (primary, when available): the result's own
       evidence chunk(s) were already scored against the user's query by
       the preprocessor's evidence search (see
       preprocessor._restructure_by_indices, which stores
       chunk_details[chunk_id]["score"]). This is normalized to a
       comparable [0, 1] relevance value at the source
       (qdrant_tools.search_evidence) regardless of whether that chunk was
       matched via dense-only cosine similarity or hybrid dense+sparse RRF
       fusion — the two retrieval paths produce structurally different raw
       scores (a bounded cosine value vs. an unbounded-below, rank-based
       fusion score), so this function never has to know or care which one
       fired for a given chunk. A triple is evidenced by exactly the
       chunk_id markdown_tools._slim_result pulled off its Neo4j edge, so
       looking that chunk_id up in chunk_details gives this triple a TRUE
       query-relevance score with no guessing about which node types or
       predicates "should" matter — the retrieval layer already decided
       that when it searched the query against the source text.
    2. LEXICAL OVERLAP (secondary): plain token overlap between the query
       and the result's own text (names, predicate, properties). This
       stays useful for exact terms (a literal entity name, a specific
       number) that embedding similarity can blur across a whole chunk,
       and costs nothing extra since it's derived purely from the query
       and the result itself — not from any hardcoded category mapping.

    Returns a float, roughly in [0, 2.0] but not hard-capped the way the
    old version was, since embedding_score is already normalized to
    [0, 1] at the source and doesn't need an artificial ceiling stacked on
    top of token overlap to stay sane.

    When NO chunk_details are available for a result (chunk_id/chunk_ids
    entirely missing, or the caller didn't pass chunk_details at all),
    this degrades gracefully to lexical-overlap-only scoring. With
    markdown_tools._slim_path now surfacing chunk_ids per hop, this
    applies equally to triples and paths — neither is structurally locked
    out of the embedding signal anymore.
    """
    clean_query = query_tokens - _NOISE_TOKENS

    # ---- Signal 1: embedding similarity via evidenced chunk(s) ----------
    embedding_score = 0.0
    chunk_ids = _result_chunk_ids(result)
    if chunk_ids and chunk_details:
        chunk_scores = [
            chunk_details[cid]["score"]
            for cid in chunk_ids
            if cid in chunk_details and isinstance(chunk_details[cid].get("score"), (int, float))
        ]
        if chunk_scores:
            # A triple can (rarely) be evidenced by more than one chunk_id
            # if it was independently surfaced by multiple worker specs
            # with different evidence — take the best, not the average, so
            # the strongest real evidence for this triple wins rather than
            # being diluted by a weaker secondary source.
            embedding_score = max(chunk_scores)

    # ---- Signal 2: lexical overlap (query-derived, not category-derived) -
    flat_text = _flatten_text(result)
    result_tokens = _tokenize(flat_text) - _NOISE_TOKENS
    token_overlap = (
        len(clean_query & result_tokens) / max(len(clean_query), 1)
        if clean_query else 0.0
    )

    # For evidence chunk results — same lexical-only treatment as before,
    # since these results ARE the evidence text itself; the embedding
    # signal above doesn't add anything beyond what token_overlap already
    # measures here, and these results may not carry a chunk_id in the
    # same field name.
    if "evidence_quote" in result or "evidence_quote_en" in result:
        quote_tokens = _tokenize(
            result.get("evidence_quote_en", "") + " " + result.get("evidence_quote", "")
        ) - _NOISE_TOKENS
        quote_overlap = len(clean_query & quote_tokens) / max(len(clean_query), 1) if clean_query else 0.0
        return token_overlap + quote_overlap

    # For path/chain results — chunk_ids (one per hop, deduplicated) are
    # now surfaced by markdown_tools._slim_path, so paths get the same
    # embedding-similarity signal triples do, not lexical-only overlap.
    # embedding_score above already covers this (it's computed from
    # _result_chunk_ids, which now reads path_data["chunk_ids"] too) — this
    # branch only adds the path-specific lexical signal: overlap against
    # the chain's own node names, which still catches exact-term matches
    # (a literal place or entity name in the query) that a chunk-level
    # embedding score can blur.
    path_data = _unwrap_path(result)
    if path_data is not None:
        chain = path_data.get("chain", [])
        chain_text = " ".join(
            str(n.get("name", "")) if isinstance(n, dict) else str(n)
            for n in chain
        )
        chain_tokens = _tokenize(chain_text) - _NOISE_TOKENS
        chain_overlap = len(clean_query & chain_tokens) / max(len(clean_query), 1) if clean_query else 0.0
        lexical_signal = max(token_overlap, chain_overlap)
        return embedding_score + lexical_signal

    # Triple case: combine both signals. Embedding similarity is weighted
    # as the primary term since it's a real measurement of query relevance;
    # lexical overlap (capped contribution) rescues exact-term matches the
    # embedding may have blurred, without letting raw keyword stuffing
    # alone dominate the score the way the old token_overlap-as-equal-term
    # design allowed.
    name_tokens = (
        _tokenize(result.get("subject", {}).get("name", "") if isinstance(result.get("subject"), dict) else "")
        | _tokenize(result.get("object", {}).get("name", "") if isinstance(result.get("object"), dict) else "")
        | _tokenize(result.get("predicate", "") if isinstance(result.get("predicate"), str) else "")
    ) - _NOISE_TOKENS
    name_overlap = len(clean_query & name_tokens) / max(len(clean_query), 1) if clean_query else 0.0

    lexical_signal = max(token_overlap, name_overlap)

    return embedding_score + lexical_signal


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
    """Render 'raw name -> (linked) -> canonical name [canonical_en]' lines
    for one result.

    entity_details / predicate_details come straight from the preprocessor
    (state["preprocessor_entity_details"] / state["preprocessor_predicate_details"]):
    canonical_id -> {"name": <canonical name>, "name_en": <English
    translation, if any>, "raw_names": [<raw surface forms>], "score":
    <float>}. A result's subject/object only carry canonical_id +
    canonical name (that's all Neo4j nodes store) — this looks up the SAME
    canonical_id in entity_details to recover every raw surface form that
    resolved to it, giving the synthesizer the link the preprocessor
    already established between what was actually extracted from text and
    what the graph calls it.

    The "[canonical_en]" suffix is appended ONLY when the canonical record
    actually has a name_en — most registry entries won't, since this
    depends on whether the live Qdrant schema populates that field. There
    is no raw-side English equivalent rendered (e.g. "raw [raw_en]") since
    raw names come from evidence_registry's plain extracted strings, which
    have no parallel translation field anywhere in the pipeline — only
    curated canonical entries plausibly carry one.

    One line per (raw_name, canonical_name) pair, deduplicated across a
    single result (a triple touches up to 3 canonical things: subject,
    predicate, object).
    """
    lines: List[str] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    def _format_canonical(canonical_name: str, canonical_name_en: str = "") -> str:
        """'canonical name [canonical_en]' when name_en exists, else just 'canonical name'."""
        if canonical_name_en:
            return f"{canonical_name} [{canonical_name_en}]"
        return canonical_name

    def _add_for(canonical_id: str, details: Dict[str, Any]) -> None:
        info = details.get(canonical_id)
        if not info:
            return
        canonical_name = info.get("name", "")
        canonical_display = _format_canonical(canonical_name, info.get("name_en", ""))
        for raw_name in info.get("raw_names", []):
            if not raw_name or raw_name == canonical_name:
                # Skip when the raw form and canonical form are identical —
                # there is nothing to link, it would just be noise.
                continue
            # Dedup key uses the raw canonical name (not the display
            # string with the [en] suffix) — the suffix is cosmetic, not
            # part of identity, so two results linking the same
            # (raw_name, canonical_name) pair still dedupe correctly
            # regardless of whether name_en is present.
            pair = (raw_name, canonical_name)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                lines.append(f"{raw_name} -> (linked) -> {canonical_display}")

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
                predicate_display = _format_canonical(predicate_name, info.get("name_en", ""))
                for raw_name in info.get("raw_names", []):
                    if raw_name and raw_name != predicate_name:
                        pair = (raw_name, predicate_name)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            lines.append(f"{raw_name} -> (linked) -> {predicate_display}")
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
    chunk_details: Optional[Dict[str, Any]] = None,
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
        chunk_details: state["preprocessor_chunk_details"] — chunk_id ->
            {"score", "community_id", "quote"}, produced by
            preprocessor._restructure_by_indices /
            preprocessor._all_from_sources. "score" is a normalized [0, 1]
            relevance value between the user's (expanded) query and that
            chunk — qdrant_tools.search_evidence computes this during
            evidence search, and normalizes it at the source regardless of
            whether the underlying Qdrant retrieval was dense-only (a true
            cosine similarity) or hybrid dense+sparse RRF fusion (a
            rank-based score on a completely different scale, rescaled to
            [0, 1] before it ever reaches this dict — see that function's
            docstring for why mixing the two scales directly would silently
            produce meaningless rankings). This is what _score_result now
            uses as its primary relevance signal instead of any hardcoded
            node-type or predicate weight table. Distinct from chunk_texts,
            which carries the fetched S3 text but drops the score field.
        max_results: Hard cap on unique results passed to the synthesizer

    Returns:
        {"filepath": <path to aggregated .md>, "result_count": <int>}
    """
    entity_details = entity_details or {}
    predicate_details = predicate_details or {}
    chunk_texts = chunk_texts or {}
    community_labels = community_labels or {}
    chunk_details = chunk_details or {}

    query_tokens = _tokenize(user_query)
    if not chunk_details:
        print(
            "  [Aggregator] No chunk_details provided — scoring will fall "
            "back to lexical overlap only. Pass state['preprocessor_chunk_details'] "
            "to enable embedding-similarity-based ranking."
        )

    seen: Dict[Tuple, Dict[str, Any]] = {}

    for filepath in markdown_files:
        for result in _extract_results_from_file(filepath):
            key = _result_key(result)
            score = _score_result(result, query_tokens, chunk_details)

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
    MAX_RELATIONSHIPS_PER_COMMUNITY = 50

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