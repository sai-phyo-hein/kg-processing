"""Check Qdrant database contents."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

collections = ["entity_registry", "predicate_registry", "ontology_registry"]

for collection_name in collections:
    print(f"\n{'='*60}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}")

    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"Points count: {collection_info.points_count}")
        print(f"Indexed vectors: {collection_info.indexed_vectors_count}")

        # Get some sample points
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
        )

        if points:
            print(f"\nSample points ({len(points)}):")
            for i, point in enumerate(points[:5], 1):
                payload = point.payload or {}
                print(f"\n{i}. ID: {point.id}")
                print(f"   Name: {payload.get('name', 'N/A')}")
                print(f"   Type: {payload.get('type', 'N/A')}")
                print(f"   Is Canonical: {payload.get('is_canonical', 'N/A')}")
                print(f"   Synonyms: {payload.get('synonyms', [])}")
                print(f"   Source Chunk: {payload.get('source_chunk', 'N/A')}")
        else:
            print("No points found in collection")

    except Exception as e:
        print(f"Error accessing collection: {e}")

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for collection_name in collections:
    try:
        collection_info = client.get_collection(collection_name)
        print(f"{collection_name}: {collection_info.points_count} points")
    except Exception as e:
        print(f"{collection_name}: Error - {e}")
