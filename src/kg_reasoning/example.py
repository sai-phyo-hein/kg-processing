"""Example usage of the KG Reasoning Engine."""

import os
from kg_reasoning.main import KGReasoningEngine


def main():
    """Example usage of the KG reasoning engine."""

    # Check environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([qdrant_url, qdrant_api_key, neo4j_uri, neo4j_user, neo4j_password]):
        print("Please set the following environment variables:")
        print("- QDRANT_URL")
        print("- QDRANT_API_KEY")
        print("- NEO4J_URI")
        print("- NEO4J_USER (or NEO4J_USERNAME)")
        print("- NEO4J_PASSWORD")
        return

    # Create reasoning engine
    engine = KGReasoningEngine(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        similarity_threshold=0.75,
    )

    # Example queries
    queries = [
        "What is the relationship between Nvidia and AI?",
        "How does debt affect company performance?",
        "What are the main causes of market volatility?",
    ]

    # Process each query
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        result = engine.query(query)

        print(f"\nAnswer:")
        print(result["answer"])

        if result["status"] == "error":
            print(f"\nError: {result['error']}")

    # Close connections
    engine.close()


if __name__ == "__main__":
    main()
