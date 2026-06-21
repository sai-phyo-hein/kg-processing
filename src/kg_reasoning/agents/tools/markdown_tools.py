"""Markdown tools for multi-agent reasoning system.

Provides LangChain tools for writing and reading query results in markdown format.
Used to store intermediate results and enable answer synthesis.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool


class MarkdownToolsManager:
    """Manager for markdown I/O operations."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize markdown tools manager.

        Args:
            output_dir: Directory for markdown output files
        """
        if output_dir is None:
            # Default to output directory in project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.output_dir = project_root / "output" / "query_results"
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sentence rendering ──────────────────────────────────────────────
    # All result shapes are rendered as human-readable sentences so the
    # synthesizer reads structured text rather than raw JSON. The LLM never
    # needs to parse JSON fields — it reads sentences like:
    #
    #   [Water User Group] --(IMPACT_ON)--> [Health Impact of Tap Water Users]
    #     community: หมู่-1_village-1  chunk: 5
    #
    #   PATH: [A] --(CAUSES)--> [B] --(LEADS_TO)--> [C]
    #
    #   EVIDENCE: "original source quote here"  (EN: "english translation")
    #     entities: X, Y  predicates: CAUSES, LEADS_TO
    #
    # Each renderer returns a plain string. write_query_results calls the
    # right renderer for each result shape and writes one sentence block
    # per result — no JSON fences in the output.

    @staticmethod
    def _render_triple(result: Dict[str, Any]) -> str:
        """Render a subject-predicate-object triple as a sentence."""
        subj = result.get("subject") or {}
        obj  = result.get("object") or {}
        pred = result.get("predicate", "?")
        community = result.get("community_id", "")
        chunk_id  = result.get("chunk_id", "")
        properties = result.get("properties") or {}

        subj_name = subj.get("name", "?") if isinstance(subj, dict) else str(subj)
        subj_type = subj.get("type", "")  if isinstance(subj, dict) else ""
        obj_name  = obj.get("name", "?")  if isinstance(obj, dict)  else str(obj)
        obj_type  = obj.get("type", "")   if isinstance(obj, dict)  else ""

        subj_str = f"[{subj_name}]" + (f" ({subj_type})" if subj_type else "")
        obj_str  = f"[{obj_name}]"  + (f" ({obj_type})"  if obj_type  else "")

        line = f"{subj_str} --({pred})--> {obj_str}"

        meta_parts = []
        if community:
            meta_parts.append(f"community: {community}")
        if chunk_id:
            meta_parts.append(f"chunk: {chunk_id}")
        if meta_parts:
            line += "\n  " + "  ".join(meta_parts)

        # Any other edge property — attribute, causal_link/weight,
        # temporal_link, etc. — rendered as its own line so it's visible to
        # a human/LLM reader without guessing a fixed property name.
        if properties:
            prop_parts = [f"{k}: {v}" for k, v in properties.items() if v is not None]
            if prop_parts:
                line += "\n  " + "  ".join(prop_parts)

        return line

    @staticmethod
    def _render_path(result: Dict[str, Any]) -> str:
        """Render a path chain as a sentence: [A] --(P)--> [B] --(Q)--> [C]."""
        path_data = result.get("path") or result.get("p") or result
        chain = path_data.get("chain", []) if isinstance(path_data, dict) else []
        community_ids = path_data.get("community_ids", []) if isinstance(path_data, dict) else []

        if not chain:
            return "[empty path]"

        parts = []
        edge_notes = []
        for item in chain:
            if isinstance(item, dict) and "predicate" in item:
                # Relationship entry from _slim_path: {"predicate": ...,
                # "properties": {...}?}. Render the predicate inline like
                # the old plain-string format, and collect any extra
                # properties as a footnote rather than breaking the
                # arrow-chain line with embedded key:value text.
                parts.append(f"--({item['predicate']})-->")
                props = item.get("properties") or {}
                if props:
                    prop_str = ", ".join(f"{k}: {v}" for k, v in props.items() if v is not None)
                    if prop_str:
                        edge_notes.append(f"[{item['predicate']}] {prop_str}")
            elif isinstance(item, dict):
                name = item.get("name", "?")
                typ  = item.get("type", "")
                parts.append(f"[{name}]" + (f" ({typ})" if typ else ""))
            elif isinstance(item, str):
                # Backward-compatible: older slim format emitted a plain
                # predicate string between nodes.
                parts.append(f"--({item})-->")
            else:
                parts.append(str(item))

        line = " ".join(parts)
        if community_ids:
            line += f"\n  communities: {', '.join(community_ids)}"
        if edge_notes:
            line += "\n  " + "  ".join(edge_notes)
        return f"PATH: {line}"

    @staticmethod
    def _render_evidence(result: Dict[str, Any]) -> str:
        """Render an evidence chunk (from Qdrant or S3) as a sentence block."""
        quote    = result.get("evidence_quote", "")
        quote_en = result.get("evidence_quote_en", "")
        entities  = result.get("entities", [])
        predicates = result.get("predicates", [])
        community = result.get("community_id", "")
        chunk_id  = result.get("chunk_id", "")

        lines = []

        # Primary quote — prefer English translation when available
        if quote_en:
            lines.append(f'EVIDENCE (EN): "{quote_en}"')
            if quote and quote != quote_en:
                lines.append(f'EVIDENCE (original): "{quote}"')
        elif quote:
            lines.append(f'EVIDENCE: "{quote}"')

        # S3 full chunk text
        text = result.get("text", "")
        if text and text != quote:
            # Truncate very long S3 chunks for display
            display = text if len(text) <= 600 else text[:600] + "…"
            lines.append(f'SOURCE TEXT: "{display}"')

        meta = []
        if entities:
            meta.append(f"entities: {', '.join(entities)}")
        if predicates:
            meta.append(f"predicates: {', '.join(predicates)}")
        if community:
            meta.append(f"community: {community}")
        if chunk_id:
            meta.append(f"chunk: {chunk_id}")
        if meta:
            lines.append("  " + "  |  ".join(meta))

        return "\n".join(lines) if lines else "[empty evidence]"

    @staticmethod
    def _render_result(result: Dict[str, Any]) -> str:
        """Dispatch to the right renderer based on result shape."""
        # Triple shape: subject / predicate / object
        if (
            "subject" in result
            and "predicate" in result
            and "object" in result
            and isinstance(result.get("subject"), dict)
            and isinstance(result.get("object"), dict)
        ):
            return MarkdownToolsManager._render_triple(result)

        # Path shape
        if "path" in result or "p" in result or (
            isinstance(result.get("chain"), list)
        ):
            return MarkdownToolsManager._render_path(result)

        # Evidence chunk shape (from Qdrant or S3)
        if "evidence_quote" in result or "evidence_quote_en" in result or "text" in result:
            return MarkdownToolsManager._render_evidence(result)

        # Generic fallback: render key: value lines
        lines = []
        for k, v in result.items():
            if isinstance(v, dict):
                v_str = ", ".join(f"{kk}: {vv}" for kk, vv in v.items() if vv)
            elif isinstance(v, list):
                v_str = ", ".join(str(i) for i in v)
            else:
                v_str = str(v) if v is not None else ""
            if v_str:
                lines.append(f"{k}: {v_str}")
        return "\n".join(lines) if lines else "[empty result]"

    # Legacy slim methods kept so any external callers don't break —
    # write_query_results no longer calls these internally.
    @staticmethod
    def _slim_node(value: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in value.get("properties", {}).items()
                if k in ("name", "canonical_id", "type")}

    @staticmethod
    def _slim_rel_type(value: Dict[str, Any]) -> str:
        return value.get("type_name") or value.get("type", "")

    @staticmethod
    def _slim_rel_community(value: Dict[str, Any]) -> Optional[str]:
        return value.get("properties", {}).get("community_id")

    @staticmethod
    def _slim_rel_chunk_id(value: Dict[str, Any]) -> Optional[Any]:
        """Same extraction pattern as _slim_rel_community, for chunk_id.

        Exists so _slim_path can surface each hop's chunk_id as its own
        top-level field — exactly how _slim_result already does for
        triples (see its `chunk_id = (rel.get("properties") or
        {}).get("chunk_id")` line) — instead of relying on
        _slim_rel_properties, which deliberately excludes chunk_id from
        the generic "properties" passthrough to avoid duplicating it
        there. Without this, a path hop's chunk_id had no top-level home
        and _slim_path silently dropped it, leaving aggregator.scoreResult
        with no way to look up that hop's evidence chunk in chunkDetails
        and therefore no embedding-similarity signal for any path result.
        """
        return value.get("properties", {}).get("chunk_id")

    # Properties excluded from the surfaced "properties" dict:
    #   - community_id, chunk_id: already surfaced as their own top-level
    #     fields elsewhere in the slimmed result, so kept out here to avoid
    #     duplication.
    #   - updated_at, label, status: graph-builder bookkeeping rather than
    #     semantic content. In observed data, updated_at carries the literal
    #     unevaluated string "datetime()" (not an actual timestamp — a bug
    #     upstream in whatever wrote it, not something to surface as if it
    #     were real temporal data), label is an internal edge-category tag
    #     duplicating the predicate's own type, and status is a lifecycle
    #     flag ("Current") rather than a fact about the relationship. None
    #     of these are the attribute/causal-link/temporal-link content this
    #     method exists to preserve.
    _REL_PROPERTY_EXCLUDE = {
        "community_id", "chunk_id",
        "updated_at", "label", "status",
    }

    @classmethod
    def _slim_rel_properties(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Return every relationship property NOT already broken out
        separately (community_id, chunk_id) and not graph-builder
        bookkeeping (updated_at, label, status) — e.g. attribute,
        causal_link/weight, temporal_link, or any other genuine edge
        property the graph builder wrote, whatever its name. This is what
        carries predicate attributes, causal-link weights, and temporal
        links through to the aggregator; without it, only the predicate
        type name and community_id survived past this slimming step, and
        any other edge property was silently dropped before a worker's
        markdown file was even written.

        Deliberately unconditional beyond the exclude list above — edges
        carry these properties inconsistently (some triples have a causal
        weight, some don't; field names aren't guaranteed to match an
        orchestrator-requested key exactly), so this passes through
        whatever actually exists rather than guessing names.
        """
        props = value.get("properties", {}) or {}
        return {
            k: v for k, v in props.items()
            if k not in cls._REL_PROPERTY_EXCLUDE
        }

    @staticmethod
    def _slim_path(value: Dict[str, Any]) -> Dict[str, Any]:
        """Slim a Neo4j path into {"chain": [...], "community_ids": [...],
        "chunk_ids": [...]}.

        Each hop's rel_entry now carries its own "chunk_id" field, the
        same top-level treatment _slim_result already gives triples. This
        is what lets aggregator._result_chunk_ids (or its TS twin) look up
        a path's REAL per-hop evidence in chunk_details for embedding-
        similarity scoring, instead of falling back to lexical-only
        overlap on node names — the same blind spot triples used to have
        before chunk_id was surfaced there.

        chunk_ids (path-level) parallels community_ids: every hop's
        chunk_id, deduplicated, in case a path's hops were evidenced by
        different chunks (a multi-hop path crossing several source
        passages). A hop without a resolvable chunk_id (older data, or a
        relationship type that genuinely has none) simply contributes
        nothing to either list — never guessed at.
        """
        nodes = value.get("nodes", [])
        rels  = value.get("relationships", [])
        if not nodes:
            return {"chain": [], "community_ids": [], "chunk_ids": []}
        chain: List[Any] = [MarkdownToolsManager._slim_node(nodes[0])]
        for i, rel in enumerate(rels):
            rel_entry: Dict[str, Any] = {
                "predicate": MarkdownToolsManager._slim_rel_type(rel),
            }
            chunk_id = MarkdownToolsManager._slim_rel_chunk_id(rel)
            if chunk_id is not None and chunk_id != "":
                rel_entry["chunk_id"] = chunk_id
            extra_props = MarkdownToolsManager._slim_rel_properties(rel)
            if extra_props:
                rel_entry["properties"] = extra_props
            chain.append(rel_entry)
            if i + 1 < len(nodes):
                chain.append(MarkdownToolsManager._slim_node(nodes[i + 1]))
        community_ids = list({
            MarkdownToolsManager._slim_rel_community(r)
            for r in rels
            if MarkdownToolsManager._slim_rel_community(r)
        })
        chunk_ids = list({
            cid for cid in (
                MarkdownToolsManager._slim_rel_chunk_id(r) for r in rels
            )
            if cid is not None and cid != ""
        })
        return {"chain": chain, "community_ids": community_ids, "chunk_ids": chunk_ids}

    @staticmethod
    def _slim_value(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        node_type = value.get("type")
        if node_type == "node":
            return MarkdownToolsManager._slim_node(value)
        if node_type == "relationship":
            entry: Dict[str, Any] = {
                "predicate":    MarkdownToolsManager._slim_rel_type(value),
                "community_id": MarkdownToolsManager._slim_rel_community(value),
            }
            extra_props = MarkdownToolsManager._slim_rel_properties(value)
            if extra_props:
                entry["properties"] = extra_props
            return entry
        if node_type == "path":
            return MarkdownToolsManager._slim_path(value)
        return {k: MarkdownToolsManager._slim_value(v) for k, v in value.items()}

    @classmethod
    def _slim_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw Neo4j result to slimmed triple/path/evidence dict."""
        subj = result.get("subject")
        rel  = result.get("r")
        obj  = result.get("object")
        if (
            subj and rel and obj
            and isinstance(subj, dict) and subj.get("type") == "node"
            and isinstance(rel,  dict) and rel.get("type")  == "relationship"
            and isinstance(obj,  dict) and obj.get("type")  == "node"
        ):
            community_id = result.get("community_id") or cls._slim_rel_community(rel)
            chunk_id = (rel.get("properties") or {}).get("chunk_id")
            slimmed: Dict[str, Any] = {
                "subject":      cls._slim_node(subj),
                "predicate":    cls._slim_rel_type(rel),
                "object":       cls._slim_node(obj),
                "community_id": community_id,
                "chunk_id":     chunk_id,
            }
            # Any other edge property — attribute, causal_link/weight,
            # temporal_link, or anything else the graph builder wrote —
            # passed through unconditionally rather than dropped. Only
            # added when non-empty so triples without extra properties
            # don't carry a noisy empty dict through to the aggregator.
            extra_props = cls._slim_rel_properties(rel)
            if extra_props:
                slimmed["properties"] = extra_props
            return slimmed
        if "path" in result or "p" in result:
            path_key = "path" if "path" in result else "p"
            return {path_key: cls._slim_value(result[path_key])}
        return {k: cls._slim_value(v) for k, v in result.items()}

    def write_query_results(
        self,
        strategy_name: str,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write query results to a markdown file.

        Raw Neo4j results are slimmed before writing: embedding vectors, sparse
        index arrays, raw source text, and internal integer IDs are stripped.
        This keeps each result row to fields that matter for synthesis
        (names, canonical_ids, labels, predicate type, community_id) and avoids
        exhausting the synthesizer context window.

        Args:
            strategy_name: Name of the query strategy
            query: The Cypher query executed
            results: List of query results (raw from Neo4j)
            metadata: Optional metadata about the query

        Returns:
            Path to the created markdown file
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() else "_" for c in strategy_name)
        filename = f"{timestamp}_{safe_name}.md"
        filepath = self.output_dir / filename

        # Build markdown content
        lines = []
        lines.append(f"# Query Results: {strategy_name}\n")
        lines.append(f"**Timestamp:** {datetime.now().isoformat()}\n")

        if metadata:
            lines.append("## Metadata\n")
            for key, value in metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("\n")

        lines.append("## Query\n")
        lines.append("```cypher")
        lines.append(query)
        lines.append("```\n")

        lines.append(f"## Results ({len(results)} records)\n")

        if not results:
            lines.append("*No results found*\n")
        elif results and "error" in results[0]:
            lines.append("### Error\n")
            lines.append(f"```\n{results[0]['error']}\n```\n")
        else:
            slimmed = [self._slim_result(r) for r in results]
            for i, result in enumerate(slimmed, 1):
                lines.append(f"### Result {i}\n")
                lines.append("```json")
                lines.append(json.dumps(result, indent=2, ensure_ascii=False))
                lines.append("```\n")

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return str(filepath)

    def read_query_results(
        self,
        filepath: Optional[str] = None,
        max_chars_per_file: int = 8000,
        max_total_chars: int = 40000,
    ) -> str:
        """Read query results from markdown file(s).

        Args:
            filepath: Optional specific file path to read (if None, reads all recent files)
            max_chars_per_file: Maximum characters to read per file (avoids oversized context)
            max_total_chars: Maximum total characters across all files

        Returns:
            Markdown content from file(s)
        """
        if filepath:
            # Read specific file
            path = Path(filepath)
            if not path.exists():
                return f"Error: File not found: {filepath}"

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file] + "\n\n*[Content truncated for length]*"
            return content
        else:
            # Read all markdown files in output directory
            md_files = sorted(self.output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

            if not md_files:
                return "No query result files found."

            # Combine recent files (up to 10), respecting size limits
            combined = []
            total_chars = 0
            for md_file in md_files[:10]:
                if total_chars >= max_total_chars:
                    combined.append(f"\n*[Remaining files omitted — size limit reached]*\n")
                    break
                header = f"\n---\n## File: {md_file.name}\n---\n"
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                if len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n\n*[Content truncated for length]*"
                remaining = max_total_chars - total_chars
                if len(content) > remaining:
                    content = content[:remaining] + "\n\n*[Content truncated for length]*"
                combined.append(header + content)
                total_chars += len(header) + len(content)

            return "\n".join(combined)

    def list_query_results(self) -> List[Dict[str, Any]]:
        """List all query result files.

        Returns:
            List of file information dictionaries
        """
        md_files = sorted(
            self.output_dir.glob("*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        files_info = []
        for md_file in md_files:
            stat = md_file.stat()
            files_info.append({
                "filename": md_file.name,
                "path": str(md_file),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return files_info


# Global manager instance
_manager = None


def _get_manager() -> MarkdownToolsManager:
    """Get or create global MarkdownToolsManager instance."""
    global _manager
    if _manager is None:
        _manager = MarkdownToolsManager()
    return _manager


@tool
def write_query_results(
    strategy_name: str,
    query: str,
    results: str,
    metadata: str = "{}",
) -> str:
    """Write query results to a markdown file for later synthesis.

    Use this tool to store the results of your Neo4j queries in a structured markdown
    format. The answer synthesizer will read these files to generate the final answer.

    Args:
        strategy_name: Name describing the query strategy (e.g., "direct_entities", "expanded_graph")
        query: The Cypher query that was executed
        results: JSON string of query results from Neo4j
        metadata: JSON string with additional metadata (default: "{}")

    Returns:
        Path to the created markdown file
    """
    manager = _get_manager()

    # Parse inputs
    try:
        results_list = json.loads(results)
    except json.JSONDecodeError:
        results_list = [{"error": "Failed to parse results", "raw": results}]

    try:
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        metadata_dict = {}

    filepath = manager.write_query_results(strategy_name, query, results_list, metadata_dict)
    return f"Results written to: {filepath}"


@tool
def read_query_results(filepath: str = "") -> str:
    """Read query results from markdown files.

    Use this tool to read previously stored query results. If no filepath is provided,
    it will return all recent query results for comprehensive answer synthesis.

    Args:
        filepath: Optional specific file path to read (empty to read all recent files)

    Returns:
        Markdown content containing query results
    """
    manager = _get_manager()
    return manager.read_query_results(filepath if filepath else None)


@tool
def list_query_results() -> str:
    """List all available query result files.

    Use this tool to see what query result files are available before reading them.

    Returns:
        JSON string with list of files and their metadata
    """
    manager = _get_manager()
    files = manager.list_query_results()
    return json.dumps(files, indent=2)