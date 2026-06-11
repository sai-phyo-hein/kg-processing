#!/usr/bin/env python3
"""
Check self-linked triples in a refined_triples.json file.

A self-linked triple is one where subject.canonical_id == object.canonical_id,
making the triple point from an entity back to itself.

Usage:
    uv run python helpers/check_self_linked_triples.py <path_to_refined_triples.json>

Example:
    uv run python helpers/check_self_linked_triples.py output/file_triples_refined.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def find_self_linked(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return all triples where subject.canonical_id == object.canonical_id."""
    self_linked: List[Dict[str, Any]] = []

    for chunk in data.get("chunks", []):
        chunk_id = chunk.get("chunk_id", "?")
        for triple in chunk.get("triples", []):
            subj = triple.get("subject", {})
            obj = triple.get("object", {})
            subj_cid = subj.get("canonical_id")
            obj_cid = obj.get("canonical_id")

            if subj_cid and obj_cid and subj_cid == obj_cid:
                self_linked.append({
                    "chunk_id": chunk_id,
                    "canonical_id": subj_cid,
                    "subject_name": subj.get("name_en") or subj.get("name"),
                    "subject_original": subj.get("original_name"),
                    "object_name": obj.get("name_en") or obj.get("name"),
                    "object_original": obj.get("original_name"),
                    "predicate": triple.get("predicate"),
                    "original_predicate": triple.get("original_predicate"),
                    "subject_label": subj.get("label"),
                    "object_label": obj.get("label"),
                })

    return self_linked


def print_report(data: Dict[str, Any], self_linked: List[Dict[str, Any]]) -> None:
    total_triples = data.get("total_triples", 0)
    count = len(self_linked)

    print("=" * 80)
    print("SELF-LINKED TRIPLES REPORT")
    print("=" * 80)
    print(f"\n📁 File: {data.get('source_file', 'N/A')}")
    print(f"📊 Total triples: {total_triples}")
    print(f"🔗 Self-linked triples: {count}")
    if total_triples > 0:
        print(f"   ({count / total_triples * 100:.2f}% of all triples)")
    print()

    if count == 0:
        print("✅ No self-linked triples found.")
        print("=" * 80)
        return

    # Group by canonical_id
    by_canonical: Dict[str, List[Dict[str, Any]]] = {}
    for t in self_linked:
        cid = t["canonical_id"]
        by_canonical.setdefault(cid, []).append(t)

    print("-" * 80)
    print(f"GROUPS BY CANONICAL ENTITY ({len(by_canonical)} unique)")
    print("-" * 80)

    # Sort groups by size descending
    for cid, group in sorted(by_canonical.items(), key=lambda x: -len(x[1])):
        entity_name = group[0]["subject_name"]
        print(f"\n  🏷️  {entity_name}  (canonical_id: {cid[:8]}…)")
        print(f"      Self-linked triples: {len(group)}")
        for t in group:
            subj_orig = t["subject_original"] or t["subject_name"]
            obj_orig = t["object_original"] or t["object_name"]
            pred = t["original_predicate"] or t["predicate"]
            s_label = t["subject_label"] or ""
            o_label = t["object_label"] or ""
            print(f"      • [{t['chunk_id']}] "
                  f"{subj_orig} ({s_label}) ──{pred}──▶ {obj_orig} ({o_label})")

    # Summary table
    print()
    print("-" * 80)
    print("SUMMARY TABLE")
    print("-" * 80)
    print(f"{'Canonical Entity':<50} {'Count':>6}")
    print("-" * 80)
    for cid, group in sorted(by_canonical.items(), key=lambda x: -len(x[1])):
        name = (group[0]["subject_name"] or "???")[:48]
        print(f"{name:<50} {len(group):>6}")
    print("-" * 80)
    print(f"{'TOTAL':<50} {count:>6}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python helpers/check_self_linked_triples.py <refined_triples.json>")
        print("\nExample:")
        print("  uv run python helpers/check_self_linked_triples.py output/file_triples_refined.json")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    self_linked = find_self_linked(data)
    print_report(data, self_linked)


if __name__ == "__main__":
    main()
