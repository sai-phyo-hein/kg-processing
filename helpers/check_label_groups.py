"""Check label groups in Qdrant against label_group_config.yaml."""

import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


def load_label_group_config(config_path: Path) -> set:
    """Load label groups from the YAML config file.
    
    Args:
        config_path: Path to label_group_config.yaml
        
    Returns:
        Set of group names from the config
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # The config is a dict where keys are the group names
    return set(config.keys()) if config else set()


def get_registry_collection_name(registry_info_dir: Path) -> str:
    """Get the Qdrant collection name for label_registry.
    
    Args:
        registry_info_dir: Path to registry_info directory
        
    Returns:
        Collection name for label registry
    """
    label_registry_file = registry_info_dir / "label_registry.json"
    
    if not label_registry_file.exists():
        print(f"Warning: {label_registry_file} not found, using default 'label_registry'")
        return "label_registry"
    
    with open(label_registry_file, 'r') as f:
        spec = json.load(f)
    
    return spec.get("collection_name", "label_registry")


def query_label_groups_from_qdrant(client: QdrantClient, collection_name: str) -> dict:
    """Query all labels from Qdrant and extract their groups.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the label registry collection
        
    Returns:
        Dict mapping group names to lists of label names in that group
    """
    print(f"📊 Querying collection '{collection_name}'...")
    
    groups = {}
    labels_without_group = []
    total_labels = 0
    
    # Scroll through all points in the collection
    offset = None
    limit = 100
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        for point in points:
            total_labels += 1
            payload = point.payload
            label_name = payload.get("name", "unknown")
            group = payload.get("group")
            
            if group:
                if group not in groups:
                    groups[group] = []
                groups[group].append(label_name)
            else:
                labels_without_group.append(label_name)
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    print(f"   Total labels: {total_labels}")
    print(f"   Labels with groups: {total_labels - len(labels_without_group)}")
    print(f"   Labels without group: {len(labels_without_group)}")
    
    if labels_without_group:
        print(f"\n⚠️  Labels without group field:")
        for label in sorted(labels_without_group):
            print(f"      - {label}")
    
    return groups


def compare_groups(qdrant_groups: dict, config_groups: set):
    """Compare groups from Qdrant with groups in config file.
    
    Args:
        qdrant_groups: Dict of groups from Qdrant (group -> list of labels)
        config_groups: Set of groups from config file
    """
    qdrant_group_names = set(qdrant_groups.keys())
    
    print("\n" + "="*70)
    print("📋 COMPARISON RESULTS")
    print("="*70)
    
    # Groups in config
    print(f"\n✅ Groups in config file: {len(config_groups)}")
    for group in sorted(config_groups):
        print(f"   - {group}")
    
    # Groups in Qdrant
    print(f"\n📦 Groups found in Qdrant: {len(qdrant_group_names)}")
    for group in sorted(qdrant_group_names):
        count = len(qdrant_groups[group])
        print(f"   - {group} ({count} label{'s' if count != 1 else ''})")
    
    # Groups in Qdrant but NOT in config (potential issues)
    missing_in_config = qdrant_group_names - config_groups
    if missing_in_config:
        print(f"\n❌ Groups in Qdrant but MISSING in config: {len(missing_in_config)}")
        for group in sorted(missing_in_config):
            count = len(qdrant_groups[group])
            print(f"   - {group} ({count} label{'s' if count != 1 else ''})")
            # Show a few example labels
            examples = qdrant_groups[group][:3]
            for label in examples:
                print(f"      → {label}")
            if len(qdrant_groups[group]) > 3:
                print(f"      → ... and {len(qdrant_groups[group]) - 3} more")
    else:
        print("\n✅ All Qdrant groups are defined in config")
    
    # Groups in config but NOT in Qdrant (unused groups)
    unused_in_config = config_groups - qdrant_group_names
    if unused_in_config:
        print(f"\n⚠️  Groups in config but NOT used in Qdrant: {len(unused_in_config)}")
        for group in sorted(unused_in_config):
            print(f"   - {group}")
    else:
        print("\n✅ All config groups are used in Qdrant")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Config groups:     {len(config_groups)}")
    print(f"Qdrant groups:     {len(qdrant_group_names)}")
    print(f"Missing in config: {len(missing_in_config)}")
    print(f"Unused in Qdrant:  {len(unused_in_config)}")
    print("="*70)


def main():
    """Main function to compare label groups."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "label_group_config.yaml"
    registry_info_dir = project_root / "registry_info"
    
    print("🔍 Checking label groups in Qdrant vs config file\n")
    print(f"Config file: {config_path}")
    print(f"Registry info: {registry_info_dir}\n")
    
    # Load config
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        return
    
    config_groups = load_label_group_config(config_path)
    print(f"✅ Loaded {len(config_groups)} groups from config\n")
    
    # Get collection name
    collection_name = get_registry_collection_name(registry_info_dir)
    
    # Connect to Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        print("❌ Error: QDRANT_URL not set in environment")
        return
    
    print(f"🔌 Connecting to Qdrant at {qdrant_url}...")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("✅ Connected to Qdrant\n")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return
    
    # Query Qdrant
    try:
        qdrant_groups = query_label_groups_from_qdrant(client, collection_name)
    except Exception as e:
        print(f"❌ Error querying Qdrant: {e}")
        return
    
    # Compare
    compare_groups(qdrant_groups, config_groups)


if __name__ == "__main__":
    main()
