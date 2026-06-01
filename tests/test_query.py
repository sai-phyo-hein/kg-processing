#!/usr/bin/env python3
"""
Test script for querying Qdrant evidence registry with user queries.

Usage:
    python tests/test_query.py "Your query here" [--limit 5]
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Add src to path so we can import from kg_extractor
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Import from kg_extractor utilities
from kg_extractor.utils.model_setup import get_embedding_client, OPENAI_EMBEDDING_MODEL

# Load environment variables
load_dotenv()


# Constants
EMBEDDING_MODEL = "text-embedding-3-large"  # For evidence, use text-embedding-3-large
COLLECTION_NAME = "ci_evidence_registry"
EVIDENCE_VECTOR_NAME = "evidence_quote_en"  # or "evidence_quote" for Thai


def get_embedding(query: str) -> List[float]:
    """
    Get embedding for a query using the configured embedding provider.
    
    Uses get_embedding_client() which supports OpenAI, OpenRouter, Groq, and NVIDIA
    based on EMBEDDING_PROVIDER environment variable.
    
    Args:
        query: Text to embed
        
    Returns:
        Embedding vector (3072 dimensions for text-embedding-3-large)
    """
    client = get_embedding_client()
    
    # Ensure text is not empty
    if not query.strip():
        query = " "
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    
    return response.data[0].embedding


def query_qdrant(
    query_vector: List[float],
    qdrant_client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    vector_name: str = EVIDENCE_VECTOR_NAME,
    limit: int = 10,
    score_threshold: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Query Qdrant collection with embedding vector.
    
    Args:
        query_vector: Embedding vector to search with
        qdrant_client: Initialized Qdrant client
        collection_name: Name of collection to query
        vector_name: Name of vector field to search
        limit: Maximum number of results
        score_threshold: Minimum similarity score
        
    Returns:
        List of matching results with id, score, and payload
    """
    try:
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            }
            for r in search_results.points
        ]
    
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return []


def print_results(query: str, results: List[Dict[str, Any]]) -> None:
    """
    Pretty print query results.
    
    Args:
        query: Original query string
        results: List of search results
    """
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    print(f"Found {len(results)} results\n")
    
    for idx, result in enumerate(results, 1):
        print(f"Result {idx}:")
        print(f"  ID: {result['id']}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Payload:")
        
        payload = result['payload']
        
        # Print key payload fields
        for key in ['evidence_quote_en', 'evidence_quote', 'chunk_id', 'source_id', 'triple_id']:
            if key in payload:
                value = payload[key]
                if key in ['evidence_quote_en', 'evidence_quote']:
                    # Truncate long text
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                print(f"    {key}: {value}")
        
        # Print any other fields
        other_fields = {k: v for k, v in payload.items() 
                       if k not in ['evidence_quote_en', 'evidence_quote', 'chunk_id', 'source_id', 'triple_id']}
        if other_fields:
            print(f"    Other fields: {list(other_fields.keys())}")
        
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query Qdrant evidence registry with semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_query.py "social capital in community"
  python tests/test_query.py "ทุนทางสังคม" --limit 10 --threshold 0.8
  python tests/test_query.py "community leaders" --vector evidence_quote
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Query string to search for"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results (default: 5)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Minimum similarity score threshold (default: 0.1)"
    )
    
    parser.add_argument(
        "--vector",
        type=str,
        choices=["evidence_quote_en", "evidence_quote"],
        default="evidence_quote_en",
        help="Vector field to search (default: evidence_quote_en for English)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})"
    )
    
    args = parser.parse_args()
    
    # Get configuration from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    if not qdrant_url:
        print("Error: QDRANT_URL not found in environment")
        sys.exit(1)
    
    try:
        # Step 1: Get embedding for query
        print(f"Embedding query with {EMBEDDING_MODEL} (provider: {embedding_provider})...")
        query_embedding = get_embedding(args.query)
        print(f"✓ Generated {len(query_embedding)}-dimensional embedding")
        
        # Step 2: Initialize Qdrant client
        print(f"Connecting to Qdrant at {qdrant_url}...")
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        print("✓ Connected to Qdrant")
        
        # Step 3: Query Qdrant
        print(f"Searching collection '{args.collection}' (vector: {args.vector})...")
        results = query_qdrant(
            query_vector=query_embedding,
            qdrant_client=qdrant_client,
            collection_name=args.collection,
            vector_name=args.vector,
            limit=args.limit,
            score_threshold=args.threshold,
        )
        
        # Step 4: Print results
        print_results(args.query, results)
        
        if not results:
            print("No results found. Try lowering the --threshold or checking the collection name.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
