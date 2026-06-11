#!/usr/bin/env python3
"""Quick diagnostic of evidence quote lengths vs chunk sizes.

Usage:
  # Auto-resolve from input file (PDF / markdown / any stem):
  python analyze_chunk_triples_coverage.py <input_file>

  # Explicit paths (backward compatible):
  python analyze_chunk_triples_coverage.py <chunks_dir> <triples.json>
"""

import json
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def resolve_paths(arg1: str, arg2: str | None) -> tuple[Path, Path]:
    """Resolve chunks directory and triples file from CLI arguments.

    If *arg2* is None, treat *arg1* as the original input file and derive
    both paths from its stem using the pipeline naming convention:
        chunks   → output/{stem}_chunks/
        triples  → output/{stem}_triples.json
    """
    if arg2 is not None:
        return Path(arg1), Path(arg2)

    stem = Path(arg1).stem
    # If user passed the _analysis.md, strip the suffix
    if stem.endswith("_analysis"):
        stem = stem[: -len("_analysis")]

    chunks_path = OUTPUT_DIR / f"{stem}_chunks"
    triples_path = OUTPUT_DIR / f"{stem}_triples.json"
    return chunks_path, triples_path


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from directory (manifest.json) or legacy single JSON."""
    if chunks_path.is_dir():
        manifest_path = chunks_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {chunks_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_chunks = data.get("chunks", [])
        parent = manifest_path.parent
        chunks = []
        for entry in raw_chunks:
            chunk_file = parent / entry["file"]
            content = chunk_file.read_text(encoding="utf-8") if chunk_file.exists() else ""
            chunks.append({
                "chunk_id": entry.get("chunk_id", 0),
                "content": content,
                "start_line": entry.get("start_line"),
                "end_line": entry.get("end_line"),
                "file": entry.get("file"),
            })
        return chunks

    # Legacy single-file format
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def analyze_chunk(
    chunk: dict,
    triples: list[dict],
) -> dict:
    """Return a stats dict for one chunk."""
    content = chunk["content"]
    chunk_chars = len(content)
    chunk_words = len(content.split())

    evidence_quotes = [
        t.get("properties", {}).get("evidence_quote", "")
        for t in triples
        if t.get("properties", {}).get("evidence_quote")
    ]

    # Deduplicate before measuring — same quote reused by N triples
    # should only count once toward coverage.
    unique_quotes = list(dict.fromkeys(evidence_quotes))  # preserves order
    total_ev_chars = sum(len(q) for q in unique_quotes)
    total_ev_words = sum(len(q.split()) for q in unique_quotes)
    num_duplicates = len(evidence_quotes) - len(unique_quotes)
    coverage = total_ev_chars / chunk_chars * 100 if chunk_chars else 0.0

    return {
        "chunk_id": chunk["chunk_id"],
        "chunk_words": chunk_words,
        "chunk_chars": chunk_chars,
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "file": chunk.get("file"),
        "num_triples": len(triples),
        "num_evidence": len(evidence_quotes),
        "num_duplicates": num_duplicates,
        "evidence_words": total_ev_words,
        "evidence_chars": total_ev_chars,
        "unique_evidence": len(unique_quotes),
        "coverage_pct": coverage,
        "quote_lengths": sorted([len(q) for q in unique_quotes], reverse=True),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_chunk_triples_coverage.py <input_file>")
        print("  python analyze_chunk_triples_coverage.py <chunks_dir> <triples.json>")
        sys.exit(1)

    arg2 = sys.argv[2] if len(sys.argv) >= 3 else None
    chunks_path, triples_path = resolve_paths(sys.argv[1], arg2)

    # Validate paths
    if not chunks_path.exists():
        print(f"❌ Chunks not found: {chunks_path}")
        sys.exit(1)
    if not triples_path.exists():
        print(f"❌ Triples not found: {triples_path}")
        sys.exit(1)

    chunks = load_chunks(chunks_path)

    with open(triples_path, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    # Build lookup: chunk_id → triples
    triples_by_chunk: dict[int, list] = {}
    for entry in triples_data.get("chunks", []):
        triples_by_chunk.setdefault(entry["chunk_id"], []).extend(
            entry.get("triples", [])
        )

    # Analyze all chunks
    results = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        triples = triples_by_chunk.get(cid, [])
        results.append(analyze_chunk(chunk, triples))

    # ── Per-chunk details ───────────────────────────────────────────────
    for r in results:
        print(f"\n{'='*70}")
        print(f"CHUNK {r['chunk_id']}")
        print(f"{'='*70}")
        print(f"Chunk size:    {r['chunk_words']} words, {r['chunk_chars']} chars")
        if r["start_line"] and r["end_line"]:
            print(f"Lines:         {r['start_line']}–{r['end_line']}")
        if r["file"]:
            print(f"File:          {r['file']}")
        print(f"Triples:       {r['num_triples']}")
        print(f"Evidence:      {r['num_evidence']} quotes ({r['num_duplicates']} duplicates)")
        print(f"  Unique:      {r['unique_evidence']} quotes, {r['evidence_words']} words, {r['evidence_chars']} chars")
        print(f"Coverage:      {r['coverage_pct']:.1f}%")

        if r["quote_lengths"]:
            print(f"\nEvidence quote lengths (chars):")
            for j, qlen in enumerate(r["quote_lengths"][:5], 1):
                print(f"  {j}. {qlen} chars")
            if len(r["quote_lengths"]) > 5:
                print(f"  ... and {len(r['quote_lengths']) - 5} more")

    # ── Summary ─────────────────────────────────────────────────────────
    total_chunks = len(results)
    total_triples = sum(r["num_triples"] for r in results)
    total_evidence = sum(r["num_evidence"] for r in results)
    total_duplicates = sum(r["num_duplicates"] for r in results)
    total_coverage = sum(r["coverage_pct"] for r in results) / total_chunks if total_chunks else 0
    chunks_no_evidence = sum(1 for r in results if r["num_evidence"] == 0)
    chunks_low_coverage = sum(1 for r in results if 0 < r["coverage_pct"] < 30)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Chunks:              {total_chunks}")
    print(f"Total triples:       {total_triples}")
    print(f"Avg triples/chunk:   {total_triples / total_chunks:.1f}" if total_chunks else "")
    print(f"Total evidence:      {total_evidence} ({total_duplicates} duplicates)")
    print(f"Avg coverage:        {total_coverage:.1f}%")
    print(f"Chunks w/o evidence: {chunks_no_evidence}")
    print(f"Chunks < 30% cover:  {chunks_low_coverage}")

    if results:
        best = max(results, key=lambda r: r["coverage_pct"])
        worst = min(results, key=lambda r: r["coverage_pct"])
        print(f"\nBest coverage:       chunk {best['chunk_id']} ({best['coverage_pct']:.1f}%)")
        print(f"Worst coverage:      chunk {worst['chunk_id']} ({worst['coverage_pct']:.1f}%)")


if __name__ == "__main__":
    main()
