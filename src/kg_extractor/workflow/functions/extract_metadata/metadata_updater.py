"""Qdrant utilities for metadata storage and retrieval with embedding-based deduplication."""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MetadataUpdater:
    """Qdrant interface for metadata storage with embedding-based deduplication."""

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        registry_info_dir: Optional[str] = None,
    ):
        """Initialize MetadataUpdater.

        Args:
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            registry_info_dir: Directory containing registry specification files
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        # Set registry info directory
        if registry_info_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent.parent
            self.registry_info_dir = project_root / "registry_info"
        else:
            self.registry_info_dir = Path(registry_info_dir)

        # Load registry specification
        self.registry_spec = self._load_registry_spec()

        # Initialize Qdrant client
        self._init_qdrant_client()

        # Ensure collection exists
        self._ensure_collection_exists()

    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient

            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=60.0,  # Increase timeout for large operations
            )
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}")

    def _load_registry_spec(self) -> Dict[str, Any]:
        """Load metadata registry specification from JSON file.

        Returns:
            Registry specification dictionary
        """
        spec_file = self.registry_info_dir / "metadata_registry.json"
        if spec_file.exists():
            with open(spec_file, "r") as f:
                return json.load(f)
        else:
            print(f"Warning: Specification file not found: {spec_file}")
            # Use default specification
            return {
                "collection_name": "metadata_registry",
                "vector_name": "metadata",
                "vector_config": {
                    "size": 1536,
                    "distance": "Cosine"
                }
            }

    def _get_vector_name(self) -> str:
        """Get the vector name for the collection."""
        return self.registry_spec.get("vector_name", "metadata")

    def _get_vector_config(self) -> Dict[str, Any]:
        """Get the vector configuration for the collection."""
        return self.registry_spec.get("vector_config", {"size": 1536, "distance": "Cosine"})

    def _ensure_collection_exists(self):
        """Ensure metadata_registry collection exists in Qdrant."""
        from qdrant_client.http import models as qdrant_models

        collection_name = "metadata_registry"

        try:
            # Check if collection exists
            self.qdrant_client.get_collection(collection_name)
            print(f"✅ Collection exists: {collection_name}")
        except Exception:
            # Create collection with correct vector configuration
            vector_name = self._get_vector_name()
            vector_config = self._get_vector_config()

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    vector_name: qdrant_models.VectorParams(
                        size=vector_config.get("size", 1536),
                        distance=qdrant_models.Distance.COSINE,
                        on_disk=vector_config.get("on_disk", False),
                        hnsw_config=qdrant_models.HnswConfigDiff(
                            m=vector_config.get("hnsw_config", {}).get("m", 24),
                            ef_construct=vector_config.get("hnsw_config", {}).get("ef_construct", 256),
                            payload_m=vector_config.get("hnsw_config", {}).get("payload_m", 24),
                        ),
                    )
                },
                sparse_vectors_config={
                    "metadata_vector": qdrant_models.SparseVectorParams(
                        index=qdrant_models.SparseIndexParams(
                            on_disk=True
                        )
                    )
                },
                on_disk_payload=True,
            )
            print(f"✅ Created collection: {collection_name}")

    def _generate_uuid(self, unique_id: str) -> str:
        """Generate a deterministic UUID from a unique_id.

        Args:
            unique_id: Unique identifier for metadata

        Returns:
            UUID string
        """
        namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
        return str(uuid.uuid5(namespace, unique_id))

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            from openai import OpenAI

            openai_api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=openai_api_key)

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  Failed to get embedding: {e}")
            # Fallback to random vector
            import random
            return [random.random() for _ in range(1536)]

    def _create_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Create a text representation of metadata for embedding.

        Args:
            metadata: Metadata dictionary

        Returns:
            Text representation
        """
        # Create a searchable text representation of metadata
        parts = []
        
        # Document information
        if metadata.get("document_title"):
            parts.append(f"Title: {metadata['document_title']}")
        if metadata.get("document_file_name"):
            parts.append(f"File: {metadata['document_file_name']}")
        if metadata.get("document_content_type"):
            parts.append(f"Content: {metadata['document_content_type']}")
        
        # Location information
        if metadata.get("location_village"):
            parts.append(f"Village: {metadata['location_village']}")
        if metadata.get("location_moo"):
            parts.append(f"Moo: {metadata['location_moo']}")
        if metadata.get("location_country"):
            parts.append(f"Country: {metadata['location_country']}")
        
        return " | ".join(parts)

    def _query_similar_metadata(
        self,
        metadata: Dict[str, Any],
        limit: int = 5,
        score_threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Query Qdrant for similar metadata entries.

        Args:
            metadata: Metadata dictionary to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of matching metadata with scores
        """
        try:
            # Create text representation and get embedding
            metadata_text = self._create_metadata_text(metadata)
            query_vector = self._get_embedding(metadata_text)

            # Get vector name for this collection
            vector_name = self._get_vector_name()

            # Search using vector similarity
            search_results = self.qdrant_client.query_points(
                collection_name="metadata_registry",
                query=query_vector,
                using=vector_name,
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )

            # Convert to expected format
            matches = []
            for result in search_results.points:
                matches.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                })

            return matches

        except Exception as e:
            print(f"⚠️  Failed to query similar metadata: {e}")
            return []

    def insert_metadata(self, metadata: Dict[str, Any]) -> None:
        """Insert or update metadata record with deduplication.

        If similar metadata exists (based on embedding similarity), use the existing one.
        Otherwise, create a new record.

        Args:
            metadata: Metadata dictionary with required fields
        """
        from qdrant_client.http import models as qdrant_models

        unique_id = metadata.get('unique_id', '')
        
        if not unique_id:
            print("⚠️  No unique_id provided, skipping metadata save")
            return

        # Try to query for similar metadata (with fallback if it fails)
        try:
            similar_metadata = self._query_similar_metadata(metadata, limit=3, score_threshold=0.85)

            if similar_metadata:
                # Use existing metadata
                best_match = similar_metadata[0]
                print(f"🔄 Similar metadata found (score: {best_match['score']:.3f})")
                print(f"   Existing ID: {best_match['payload'].get('unique_id')}")
                print(f"   New ID: {unique_id}")
                print(f"   Using existing metadata instead of creating new record")
                return
        except Exception as e:
            # If similarity search fails, continue with insertion
            print(f"⚠️  Similarity search failed: {e}")
            print(f"   Continuing with insertion...")

        # No similar metadata found, create new record
        print(f"📝 No similar metadata found, creating new record: {unique_id}")

        # Prepare payload with defaults for missing fields
        payload = {
            'unique_id': unique_id,
            'document_title': metadata.get('document_title', ''),
            'document_file_name': metadata.get('document_file_name', ''),
            'document_file_type': metadata.get('document_file_type', ''),
            'document_total_pages': metadata.get('document_total_pages', 0),
            'document_content_type': metadata.get('document_content_type', ''),
            'location_village': metadata.get('location_village', ''),
            'location_moo': metadata.get('location_moo', ''),
            'location_country': metadata.get('location_country', ''),
        }

        # Create embedding
        metadata_text = self._create_metadata_text(metadata)
        embedding = self._get_embedding(metadata_text)

        # Generate point ID
        point_id = self._generate_uuid(unique_id)
        vector_name = self._get_vector_name()

        # Create point
        point = qdrant_models.PointStruct(
            id=point_id,
            vector={
                vector_name: embedding
            },
            payload=payload,
        )

        # Upsert to Qdrant with retry
        try:
            self.qdrant_client.upsert(
                collection_name="metadata_registry",
                points=[point],
                wait=True,  # Wait for indexing to complete
            )
            print(f"💾 Metadata saved to Qdrant: {unique_id}")
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
            raise

    def get_metadata(self, unique_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata by unique ID.

        Args:
            unique_id: Unique identifier for the metadata record

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            point_id = self._generate_uuid(unique_id)
            
            result = self.qdrant_client.retrieve(
                collection_name="metadata_registry",
                ids=[point_id],
                with_payload=True,
            )

            if result:
                return result[0].payload
            return None

        except Exception as e:
            print(f"⚠️  Failed to retrieve metadata: {e}")
            return None

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Retrieve all metadata records.

        Returns:
            List of metadata dictionaries
        """
        try:
            # Scroll through all points
            scroll_result = self.qdrant_client.scroll(
                collection_name="metadata_registry",
                limit=1000,
                with_payload=True,
            )

            points, next_page = scroll_result

            metadata_list = [point.payload for point in points]

            # Continue scrolling if there are more pages
            while next_page:
                scroll_result = self.qdrant_client.scroll(
                    collection_name="metadata_registry",
                    limit=1000,
                    with_payload=True,
                    offset=next_page,
                )
                points, next_page = scroll_result
                metadata_list.extend([point.payload for point in points])

            return metadata_list

        except Exception as e:
            print(f"⚠️  Failed to retrieve all metadata: {e}")
            return []

    def search_by_location(self, moo: str = None, village: str = None) -> List[Dict[str, Any]]:
        """Search metadata by location using filters.

        Args:
            moo: หมู่ number to search for
            village: Village name to search for

        Returns:
            List of matching metadata dictionaries
        """
        try:
            from qdrant_client.http import models as qdrant_models

            # Build filter conditions
            conditions = []

            if moo:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key="location_moo",
                        match=qdrant_models.MatchText(text=moo),
                    )
                )

            if village:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key="location_village",
                        match=qdrant_models.MatchText(text=village),
                    )
                )

            if not conditions:
                return self.get_all_metadata()

            # Search with filters
            search_filter = qdrant_models.Filter(
                must=conditions
            )

            scroll_result = self.qdrant_client.scroll(
                collection_name="metadata_registry",
                scroll_filter=search_filter,
                limit=1000,
                with_payload=True,
            )

            points, _ = scroll_result
            return [point.payload for point in points]

        except Exception as e:
            print(f"⚠️  Failed to search by location: {e}")
            return []

    def delete_metadata(self, unique_id: str) -> bool:
        """Delete metadata by unique ID.

        Args:
            unique_id: Unique identifier for the metadata record

        Returns:
            True if deleted, False if not found
        """
        try:
            from qdrant_client.http import models as qdrant_models
            
            point_id = self._generate_uuid(unique_id)
            
            self.qdrant_client.delete(
                collection_name="metadata_registry",
                points_selector=qdrant_models.PointIdsList(
                    points=[point_id]
                ),
            )
            
            print(f"🗑️  Metadata deleted: {unique_id}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to delete metadata: {e}")
            return False


def save_metadata(
    metadata: Dict[str, Any],
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> None:
    """Save metadata to Qdrant with deduplication.

    Args:
        metadata: Metadata dictionary
        qdrant_url: Qdrant server URL (from env if not provided)
        qdrant_api_key: Qdrant API key (from env if not provided)
    """
    updater = MetadataUpdater(qdrant_url, qdrant_api_key)
    updater.insert_metadata(metadata)


def load_metadata(
    unique_id: str,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load metadata from Qdrant.

    Args:
        unique_id: Unique identifier
        qdrant_url: Qdrant server URL (from env if not provided)
        qdrant_api_key: Qdrant API key (from env if not provided)

    Returns:
        Metadata dictionary or None
    """
    updater = MetadataUpdater(qdrant_url, qdrant_api_key)
    return updater.get_metadata(unique_id)
