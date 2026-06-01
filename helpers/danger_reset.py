#!/usr/bin/env python3
"""
DANGER: This script deletes ALL data from Qdrant and Neo4j databases.

This is a destructive operation that cannot be undone. Use with extreme caution.
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DangerReset:
    """Reset databases by deleting all data."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize the danger reset tool.

        Args:
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            neo4j_uri: Neo4j server URI (from env if not provided)
            neo4j_user: Neo4j username (from env if not provided)
            neo4j_password: Neo4j password (from env if not provided)
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

    def reset_qdrant(self, collection_type: Optional[str] = None) -> bool:
        """Delete all points from Qdrant collections (keeps collections).

        Args:
            collection_type: Specific collection type to delete (entity, predicate, 
                           label, metadata, evidence). If None, deletes from all collections.

        Returns:
            Success status
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter

            print("⚠️  Connecting to Qdrant...")
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )

            # Get all collections
            collections = client.get_collections().collections
            all_collection_names = [col.name for col in collections]

            if not all_collection_names:
                print("✅ No collections found in Qdrant")
                return True

            # Filter collections by type if specified
            if collection_type:
                # Map collection type to name pattern
                type_mapping = {
                    'entity': '_entity_registry',
                    'predicate': '_predicate_registry',
                    'label': '_label_registry',
                    'metadata': '_metadata_registry',
                    'evidence': '_evidence_registry',
                }
                
                pattern = type_mapping.get(collection_type.lower())
                if not pattern:
                    print(f"❌ Unknown collection type: {collection_type}")
                    print(f"   Valid types: {', '.join(type_mapping.keys())}")
                    return False
                
                collection_names = [name for name in all_collection_names if pattern in name]
                if not collection_names:
                    print(f"⚠️  No collections found matching type '{collection_type}'")
                    return True
                
                print(f"🎯 Targeting {collection_type} collections: {collection_names}")
            else:
                collection_names = all_collection_names
                print(f"🗑️  Found {len(collection_names)} collections: {collection_names}")

            print("⚠️  DELETING ALL POINTS FROM COLLECTIONS...")

            for collection_name in collection_names:
                # Get point count before deletion
                collection_info = client.get_collection(collection_name)
                point_count = collection_info.points_count

                if point_count > 0:
                    # Delete all points from the collection using empty filter
                    client.delete(
                        collection_name=collection_name,
                        points_selector=Filter(must=[]),
                    )
                    print(f"   ✓ Deleted {point_count} points from collection: {collection_name}")
                else:
                    print(f"   ✓ Collection {collection_name} already empty")

            if collection_type:
                print(f"✅ {collection_type.capitalize()} collections cleared successfully")
            else:
                print("✅ All Qdrant points deleted successfully (collections preserved)")
            return True

        except ImportError:
            print("❌ Error: qdrant-client is required. Install with: pip install qdrant-client")
            return False
        except Exception as e:
            print(f"❌ Error resetting Qdrant: {e}")
            return False

    def reset_neo4j(self) -> bool:
        """Delete all nodes and relationships from Neo4j.

        Returns:
            Success status
        """
        try:
            from neo4j import GraphDatabase

            print("⚠️  Connecting to Neo4j...")
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )

            with driver.session() as session:
                # Count nodes before deletion
                result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = result.single()["count"]

                # Count relationships before deletion
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = result.single()["count"]

                print(f"🗑️  Found {node_count} nodes and {rel_count} relationships")
                print("⚠️  DELETING ALL NODES AND RELATIONSHIPS...")

                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")

                # Verify deletion
                result = session.run("MATCH (n) RETURN count(n) as count")
                final_node_count = result.single()["count"]

                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                final_rel_count = result.single()["count"]

                if final_node_count == 0 and final_rel_count == 0:
                    print("✅ All Neo4j data deleted successfully")
                    return True
                else:
                    print(f"⚠️  Warning: {final_node_count} nodes and {final_rel_count} relationships remain")
                    return False

        except ImportError:
            print("❌ Error: neo4j is required. Install with: pip install neo4j")
            return False
        except Exception as e:
            print(f"❌ Error resetting Neo4j: {e}")
            return False
        finally:
            if 'driver' in locals():
                driver.close()

    def reset_all(self, collection_type: Optional[str] = None) -> bool:
        """Reset both Qdrant and Neo4j databases.

        Args:
            collection_type: Specific collection type to delete from Qdrant.
                           If None, deletes from all collections.

        Returns:
            Success status
        """
        print("\n" + "=" * 60)
        if collection_type:
            print(f"🚨 DANGER: QDRANT {collection_type.upper()} COLLECTION RESET 🚨")
        else:
            print("🚨 DANGER: DATABASE RESET 🚨")
        print("=" * 60)
        print("This will DELETE ALL DATA from:")
        if collection_type:
            print(f"  - Qdrant: {self.qdrant_url} ({collection_type} collections only)")
        else:
            print(f"  - Qdrant: {self.qdrant_url} (points only, collections preserved)")
            print(f"  - Neo4j: {self.neo4j_uri} (all nodes and relationships)")
        print("=" * 60)
        print()

        qdrant_success = self.reset_qdrant(collection_type=collection_type)
        
        # Only reset Neo4j if no specific collection type is specified
        neo4j_success = True
        if not collection_type:
            print()
            neo4j_success = self.reset_neo4j()

        print()
        print("=" * 60)
        if qdrant_success and neo4j_success:
            if collection_type:
                print(f"✅ {collection_type.upper()} COLLECTION RESET SUCCESSFULLY")
            else:
                print("✅ ALL DATABASES RESET SUCCESSFULLY")
        else:
            print("⚠️  RESET COMPLETED WITH ERRORS")
        print("=" * 60)

        return qdrant_success and neo4j_success


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Reset Qdrant and Neo4j databases")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt (use with extreme caution)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        choices=["entity", "predicate", "label", "metadata", "evidence"],
        help="Specific collection type to delete (entity, predicate, label, metadata, evidence). "
             "If not specified, deletes all Qdrant collections and Neo4j data."
    )
    args = parser.parse_args()

    print("\n" + "⚠️ " * 30)
    if args.collection:
        print(f"⚠️  DANGER: THIS WILL DELETE {args.collection.upper()} DATA  ⚠️")
    else:
        print("⚠️  DANGER: THIS WILL DELETE ALL DATA  ⚠️")
    print("⚠️ " * 30)
    print()

    # Require explicit confirmation unless --force is used
    if not args.force:
        try:
            if args.collection:
                confirmation = input(f"Type 'DELETE' to confirm you want to delete {args.collection} data: ")
            else:
                confirmation = input("Type 'DELETE' to confirm you want to delete all data: ")
            if confirmation != "DELETE":
                print("❌ Operation cancelled. No data was deleted.")
                sys.exit(1)
        except EOFError:
            print("❌ Cannot read input in non-interactive mode. Use --force to skip confirmation.")
            sys.exit(1)
    else:
        print("⚠️  --force flag detected, skipping confirmation...")

    print()
    if args.collection:
        print(f"🔄 Starting {args.collection} collection reset...")
    else:
        print("🔄 Starting database reset...")

    # Create reset tool
    reset = DangerReset()

    # Reset specified collection or all databases
    success = reset.reset_all(collection_type=args.collection)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()