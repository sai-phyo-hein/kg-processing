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

    def reset_qdrant(self) -> bool:
        """Delete all points from Qdrant collections (keeps collections).

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
            collection_names = [col.name for col in collections]

            if not collection_names:
                print("✅ No collections found in Qdrant")
                return True

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

    def reset_all(self) -> bool:
        """Reset both Qdrant and Neo4j databases.

        Returns:
            Success status
        """
        print("\n" + "=" * 60)
        print("🚨 DANGER: DATABASE RESET 🚨")
        print("=" * 60)
        print("This will DELETE ALL DATA from:")
        print(f"  - Qdrant: {self.qdrant_url} (points only, collections preserved)")
        print(f"  - Neo4j: {self.neo4j_uri} (all nodes and relationships)")
        print("=" * 60)
        print()

        qdrant_success = self.reset_qdrant()
        print()
        neo4j_success = self.reset_neo4j()

        print()
        print("=" * 60)
        if qdrant_success and neo4j_success:
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
    args = parser.parse_args()

    print("\n" + "⚠️ " * 30)
    print("⚠️  DANGER: THIS WILL DELETE ALL DATA  ⚠️")
    print("⚠️ " * 30)
    print()

    # Require explicit confirmation unless --force is used
    if not args.force:
        try:
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
    print("🔄 Starting database reset...")

    # Create reset tool
    reset = DangerReset()

    # Reset all databases
    success = reset.reset_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()