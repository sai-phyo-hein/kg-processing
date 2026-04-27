"""Check canonical_ids in refined triples."""

import json

with open("output/test_triples_refined.json", "r") as f:
    data = json.load(f)

# Count triples with canonical_ids
total_triples = 0
with_source_canonical = 0
with_target_canonical = 0
with_both_canonical = 0

for chunk in data["chunks"]:
    for triple in chunk["triples"]:
        total_triples += 1
        source_id = triple["refinement"]["subject"].get("canonical_id")
        target_id = triple["refinement"]["object"].get("canonical_id")

        if source_id:
            with_source_canonical += 1
        if target_id:
            with_target_canonical += 1
        if source_id and target_id:
            with_both_canonical += 1

print(f"Total triples: {total_triples}")
print(f"Triples with source canonical_id: {with_source_canonical}")
print(f"Triples with target canonical_id: {with_target_canonical}")
print(f"Triples with both canonical_ids: {with_both_canonical}")

# Show sample triple
sample = data["chunks"][0]["triples"][0]
print(f"\nSample triple:")
print(f"Subject: {sample['subject']['name']} (canonical_id: {sample['refinement']['subject'].get('canonical_id')})")
print(f"Predicate: {sample['predicate']}")
print(f"Object: {sample['object']['name']} (canonical_id: {sample['refinement']['object'].get('canonical_id')})")