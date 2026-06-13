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

    # Properties stripped before writing to markdown.
    # Embedding vectors (~512 floats each) and sparse index arrays are the
    # primary token-count offenders. A single triple with embeddings on
    # subject, predicate, and object can exceed 3000 tokens. Raw/source
    # text duplicates what the predicate name already conveys.
    # Node properties kept for synthesis — everything else is dropped.
    # canonical_id and name identify the entity; type describes what it is.
    _KEEP_NODE_PROPS: frozenset = frozenset([
        "name", "canonical_id", "type",
    ])

    # Relationship properties kept for synthesis.
    # type_name carries the semantic relation; community_id scopes it to a
    # village. Everything else (chunk_id, updated_at, status, label,
    # embeddings, sparse arrays, source/raw text) is indexing metadata.
    _KEEP_REL_PROPS: frozenset = frozenset([
        "community_id",
    ])

    @staticmethod
    def _slim_node(value: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only name, canonical_id, type from a node."""
        return {
            k: v for k, v in value.get("properties", {}).items()
            if k in MarkdownToolsManager._KEEP_NODE_PROPS
        }

    @staticmethod
    def _slim_rel_type(value: Dict[str, Any]) -> str:
        """Extract predicate name from a relationship."""
        return value.get("type_name") or value.get("type", "")

    @staticmethod
    def _slim_rel_community(value: Dict[str, Any]) -> Optional[str]:
        """Extract community_id from a relationship."""
        return value.get("properties", {}).get("community_id")

    @staticmethod
    def _slim_path(value: Dict[str, Any]) -> Dict[str, Any]:
        """Render a path as an ordered chain: [node, predicate, node, predicate, node, ...].

        This preserves connectivity — the synthesizer can read the chain left-to-right
        and know exactly which nodes are joined by which predicate, rather than
        receiving two separate parallel arrays of nodes and relationships.
        """
        nodes = value.get("nodes", [])
        rels  = value.get("relationships", [])
        if not nodes:
            return {"chain": []}
        chain: List[Any] = [MarkdownToolsManager._slim_node(nodes[0])]
        for i, rel in enumerate(rels):
            chain.append(MarkdownToolsManager._slim_rel_type(rel))
            if i + 1 < len(nodes):
                chain.append(MarkdownToolsManager._slim_node(nodes[i + 1]))
        community_ids = list({
            MarkdownToolsManager._slim_rel_community(r)
            for r in rels
            if MarkdownToolsManager._slim_rel_community(r)
        })
        return {"chain": chain, "community_ids": community_ids}

    @staticmethod
    def _slim_value(value: Any) -> Any:
        """Slim a single Neo4j value — used for non-triple row fields."""
        if not isinstance(value, dict):
            return value
        node_type = value.get("type")
        if node_type == "node":
            return MarkdownToolsManager._slim_node(value)
        if node_type == "relationship":
            return {
                "predicate":    MarkdownToolsManager._slim_rel_type(value),
                "community_id": MarkdownToolsManager._slim_rel_community(value),
            }
        if node_type == "path":
            return MarkdownToolsManager._slim_path(value)
        return {k: MarkdownToolsManager._slim_value(v) for k, v in value.items()}

    @classmethod
    def _slim_result(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """Slim one raw Neo4j result row.

        Detects the common subject/r/object triple shape and renders it as:
          { subject: {name, canonical_id, type},
            predicate: "REL_TYPE",
            object: {name, canonical_id, type},
            community_id: "..." }

        Path rows (multi-hop expansion) are rendered as a chain list so
        connectivity is explicit. The redundant "rels" column that
        accompanies path results is dropped — it duplicates what the chain
        already contains.

        All other shapes are slimmed field-by-field.
        """
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
            return {
                "subject":      cls._slim_node(subj),
                "predicate":    cls._slim_rel_type(rel),
                "object":       cls._slim_node(obj),
                "community_id": community_id,
            }
        # Path rows: keep only the path key (rendered as a chain), drop "rels"
        # which is the raw relationships(path) column and duplicates the chain.
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