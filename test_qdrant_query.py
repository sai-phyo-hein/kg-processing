"""Test Qdrant query functionality with real embeddings."""

import os
from dotenv import load_dotenv
from kg_extractor.utils.triple_refiner import TripleRefiner

load_dotenv()

print("🧪 Testing Qdrant query functionality...\n")

# Create refiner
refiner = TripleRefiner(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
)

# Test 1: Query empty collection (should return empty)
print("Test 1: Query empty collection")
results = refiner._query_qdrant("predicate_registry", "ACHIEVE_HIGHER_ENTERPRISE_VALUES", limit=5)
print(f"Results: {len(results)} matches")
print(f"Expected: 0 matches (empty collection)\n")

# Test 2: Upsert some predicates
print("Test 2: Upsert test predicates")
test_predicates = [
    "ACHIEVE_HIGHER_ENTERPRISE_VALUES",
    "ACHIEVE_HIGHER_VALUATIONS",
    "INCREASE_VALUATION",
    "MAXIMIZE_ENTERPRISE_VALUE",
]

for predicate in test_predicates:
    point_id = refiner._upsert_entity(
        collection_name="predicate_registry",
        name=predicate,
        entity_type="predicate",
        is_canonical=True,
        synonyms=[]
    )
    print(f"  Upserted: {predicate} -> {point_id}")

print()

# Test 3: Query for similar predicates
print("Test 3: Query for similar predicates")
query_text = "ACHIEVE_HIGHER_ENTERPRISE_VALUES"
results = refiner._query_qdrant("predicate_registry", query_text, limit=5)

print(f"Query: '{query_text}'")
print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['payload']['name']} (score: {result['score']:.4f})")

print()

# Test 4: Query for a semantically similar predicate
print("Test 4: Query for semantically similar predicate")
query_text = "ACHIEVE_HIGHER_VALUATIONS"
results = refiner._query_qdrant("predicate_registry", query_text, limit=5)

print(f"Query: '{query_text}'")
print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['payload']['name']} (score: {result['score']:.4f})")

print()

# Test 5: Check if similar predicates are matched
print("Test 5: Check if similar predicates are matched")
query_text = "INCREASE_VALUATION"
results = refiner._query_qdrant("predicate_registry", query_text, limit=5)

print(f"Query: '{query_text}'")
print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"  {i}. {result['payload']['name']} (score: {result['score']:.4f})")

print("\n✅ Test completed!")

# Close connection
refiner.qdrant_client.close()