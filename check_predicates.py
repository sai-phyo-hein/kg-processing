"""Check predicate resolution in refined triples."""

import json
from collections import Counter

with open("output/test_triples_refined.json", "r") as f:
    data = json.load(f)

# Collect all predicates
original_predicates = []
refined_predicates = []

for chunk in data["chunks"]:
    for triple in chunk["triples"]:
        original_predicates.append(triple["original_predicate"])
        refined_predicates.append(triple["predicate"])

print(f"Total triples: {len(original_predicates)}")
print(f"Unique original predicates: {len(set(original_predicates))}")
print(f"Unique refined predicates: {len(set(refined_predicates))}")

print("\n📊 Original predicates (count):")
for pred, count in Counter(original_predicates).most_common():
    print(f"  {pred}: {count}")

print("\n📊 Refined predicates (count):")
for pred, count in Counter(refined_predicates).most_common():
    print(f"  {pred}: {count}")

# Check if predicates were actually refined
changed = sum(1 for orig, ref in zip(original_predicates, refined_predicates) if orig != ref)
print(f"\n🔄 Predicates changed: {changed}/{len(original_predicates)} ({changed/len(original_predicates)*100:.1f}%)")

# Show some examples of changes
print("\n🔍 Examples of predicate changes:")
changes = [(orig, ref) for orig, ref in zip(original_predicates, refined_predicates) if orig != ref]
for orig, ref in changes[:10]:
    print(f"  {orig} → {ref}")