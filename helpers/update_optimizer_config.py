"""Update Qdrant optimizer config for all registry collections."""
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import OptimizersConfigDiff


def update_optimizer_configs():
    """Update optimizer config for all registry collections."""
    # Load environment variables
    load_dotenv()
    
    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        raise ValueError("QDRANT_URL not found in environment variables")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY not found in environment variables")
    
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    
    # Get all registry files
    registry_dir = Path(__file__).parent.parent / "registry_info"
    registry_files = list(registry_dir.glob("*_registry.json"))
    
    # Extract collection names from registry files
    collection_names = []
    for registry_file in registry_files:
        # Convert "entity_registry.json" to "ci_entity_registry"
        registry_name = registry_file.stem  # e.g., "entity_registry"
        collection_name = f"ci_{registry_name}"
        collection_names.append(collection_name)
    
    print(f"Found {len(collection_names)} registry collections:")
    for name in collection_names:
        print(f"  - {name}")
    print()
    
    # Update optimizer config for each collection
    success_count = 0
    for collection_name in collection_names:
        try:
            # Check if collection exists
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                print(f"⚠️  Collection '{collection_name}' does not exist - skipping")
                continue
            
            # Update optimizer config
            client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=100  # Build index after 100 vectors
                )
            )
            print(f"✓ Updated optimizer config for '{collection_name}'")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error updating '{collection_name}': {e}")
    
    print()
    print(f"Summary: Updated {success_count}/{len(collection_names)} collections")


if __name__ == "__main__":
    update_optimizer_configs()
