"""Main module for kg-reasoning."""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from kg_reasoning.utils.entity_extractor import EntityExtractor
from kg_reasoning.utils.qdrant_matcher import QdrantMatcher
from kg_reasoning.utils.cypher_generator import CypherGenerator
from kg_reasoning.utils.neo4j_query import Neo4jQuery
from kg_reasoning.utils.answer_synthesizer import AnswerSynthesizer
from kg_reasoning.utils.langgraph_workflow import run_langgraph_workflow


class KGReasoningEngine:
    """Main reasoning engine for knowledge graph querying.

    Uses hybrid matching approach combining keyword-based precision with
    semantic understanding for optimal knowledge graph querying.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        similarity_threshold: float = 0.75,
    ):
        """Initialize the KG reasoning engine.

        Args:
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            neo4j_uri: Neo4j server URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            llm_provider: LLM provider
            llm_model: LLM model
            similarity_threshold: Similarity threshold for entity matching
        """
        self.similarity_threshold = similarity_threshold

        # Initialize components with hybrid matching approach
        # Keyword weight: 0.7 (prioritize precision for exact terminology)
        # Semantic weight: 0.3 (allow flexibility for synonyms and context)
        self.entity_extractor = EntityExtractor(
            llm_provider,
            llm_model,
            qdrant_url,
            qdrant_api_key,
            keyword_weight=0.7,  # Prioritize precision
            semantic_weight=0.3,  # But allow semantic flexibility
        )
        self.qdrant_matcher = QdrantMatcher(qdrant_url, qdrant_api_key)
        self.cypher_generator = CypherGenerator(llm_provider, llm_model)
        self.neo4j_query = Neo4jQuery(neo4j_uri, neo4j_user, neo4j_password)
        self.answer_synthesizer = AnswerSynthesizer(llm_provider, llm_model)

    def query(
        self,
        user_query: str,
    ) -> Dict[str, Any]:
        """Process a user query and return the answer.

        Args:
            user_query: The user's natural language query

        Returns:
            Dictionary with the answer and metadata
        """
        result = {
            "original_query": user_query,
            "status": "success",
            "answer": "",
            "metadata": {},
        }

        try:
            # Step 1: Extract entities and predicates from user query
            print("🔍 Extracting entities and predicates from query...")
            extraction_result = self.entity_extractor.extract_entities(user_query)
            result["metadata"]["entity_extraction"] = extraction_result

            entities = extraction_result.get("entities", [])
            predicates = extraction_result.get("predicates", [])
            print(f"   Found {len(entities)} entities and {len(predicates)} predicates")

            # Step 2: Match entities in Qdrant (already done in entity_extractor with hybrid approach)
            # Get the matches from the extraction result
            entity_matches = extraction_result.get("qdrant_entity_matches", [])
            predicate_matches = extraction_result.get("qdrant_predicate_matches", [])
            keywords = extraction_result.get("keywords", [])

            print(f"   Hybrid matching results:")
            print(f"   - Keywords extracted: {len(keywords)}")
            print(f"   - Entity matches: {len(entity_matches)} (combined keyword + semantic)")
            print(f"   - Predicate matches: {len(predicate_matches)} (combined keyword + semantic)")

            # Show match type breakdown
            keyword_entity_matches = [m for m in entity_matches if "keyword" in m.get("match_type", [])]
            semantic_entity_matches = [m for m in entity_matches if "semantic" in m.get("match_type", [])]
            print(f"   - Entity match breakdown: {len(keyword_entity_matches)} keyword, {len(semantic_entity_matches)} semantic")

            # Store matches in metadata
            result["metadata"]["entity_matches"] = entity_matches
            result["metadata"]["predicate_matches"] = predicate_matches
            result["metadata"]["keywords"] = keywords

            # Step 3: Check if we have meaningful matches
            # Consider matches meaningful if we have at least 2 entity matches or 1 entity + 1 predicate
            has_meaningful_matches = (
                (len(entity_matches) >= 2) or
                (len(entity_matches) >= 1 and len(predicate_matches) >= 1)
            )

            if has_meaningful_matches:
                # Step 3a: Refine query with canonical entities and predicates
                print("✏️  Refining query with canonical entities and predicates...")
                refined_query = self._refine_query(user_query, entity_matches, predicate_matches)
                result["metadata"]["refined_query"] = refined_query
                print(f"   Refined query: {refined_query}")

                # Step 3b: Generate Cypher query with matched entities
                print("🔮 Generating Cypher query with matched entities...")
                cypher_result = self.cypher_generator.generate_cypher_with_entities(
                    refined_query,
                    entity_matches,
                    predicate_matches
                )
                cypher_query = cypher_result["cypher_query"]
                result["metadata"]["cypher_query"] = cypher_query
                result["metadata"]["cypher_explanation"] = cypher_result["query_explanation"]
                print(f"   Generated Cypher query: {cypher_query}")

                # Step 3c: Execute Neo4j query
                print("🕸️  Executing Neo4j query...")
                print(f"   Query: {cypher_query}")
                neo4j_results = self.neo4j_query.execute_query(cypher_query)
                result["metadata"]["neo4j_results"] = neo4j_results
                print(f"   Found {len(neo4j_results)} results")

                # Step 3d: Synthesize answer
                print("📝 Synthesizing answer...")
                answer = self.answer_synthesizer.synthesize_answer(
                    user_query,
                    neo4j_results,
                    cypher_query,
                )
                result["answer"] = answer
                print(f"   Answer synthesized")

            else:
                # Step 3b: No meaningful matches found, get high connectivity nodes first
                print("📊 No meaningful entity matches found, getting high connectivity nodes...")
                high_connectivity_nodes = self.neo4j_query.get_high_connectivity_nodes(limit=20)
                result["metadata"]["high_connectivity_nodes"] = high_connectivity_nodes
                print(f"   Found {len(high_connectivity_nodes)} high connectivity nodes")

                # Step 3c: Generate and execute query based on high connectivity nodes
                print("🔮 Generating query based on high connectivity nodes...")
                cypher_result = self.cypher_generator.generate_cypher_from_nodes(
                    user_query,
                    high_connectivity_nodes[:5]  # Use top 5 nodes
                )
                cypher_query = cypher_result["cypher_query"]
                result["metadata"]["cypher_query"] = cypher_query
                result["metadata"]["cypher_explanation"] = cypher_result["query_explanation"]
                print(f"   Generated Cypher query: {cypher_query}")

                # Step 3d: Execute Neo4j query
                print("🕸️  Executing Neo4j query...")
                print(f"   Query: {cypher_query}")
                neo4j_results = self.neo4j_query.execute_query(cypher_query)
                result["metadata"]["neo4j_results"] = neo4j_results
                print(f"   Found {len(neo4j_results)} results")

                # Step 3e: Synthesize answer
                print("📝 Synthesizing answer...")
                answer = self.answer_synthesizer.synthesize_answer(
                    user_query,
                    neo4j_results,
                    cypher_query,
                )
                result["answer"] = answer
                print(f"   Answer synthesized")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"❌ Error: {e}")

        return result

    def _refine_query(
        self,
        original_query: str,
        entity_matches: List[Dict[str, Any]],
        predicate_matches: List[Dict[str, Any]] = None,
    ) -> str:
        """Refine user query with canonical entities and predicates.

        Args:
            original_query: The original user query
            entity_matches: List of matched canonical entities
            predicate_matches: List of matched canonical predicates

        Returns:
            Refined query with canonical entities and predicates
        """
        # Simple refinement: replace entities and predicates in the query
        refined_query = original_query

        # Replace entities
        for match in entity_matches:
            original = match["payload"].get("name", "")
            canonical = match["payload"].get("name", "")  # Use the matched name as canonical
            if original and canonical:
                refined_query = refined_query.replace(original, canonical)

        # Replace predicates if available
        if predicate_matches:
            for match in predicate_matches:
                original = match["payload"].get("name", "")
                canonical = match["payload"].get("name", "")  # Use the matched name as canonical
                if original and canonical:
                    refined_query = refined_query.replace(original, canonical)

        return refined_query

    def close(self):
        """Close all connections."""
        self.neo4j_query.close()


def query_command(args) -> None:
    """Run a query against the knowledge graph."""
    print("🚀 Starting KG reasoning query...")
    print(f"❓ Query: {args.query}")
    print(f"🤖 LLM Provider: {args.llm_provider}")
    print(f"🧠 LLM Model: {args.llm_model}")
    print(f"🎯 Similarity Threshold: {args.similarity_threshold}")
    print()

    try:
        # Check for environment variables
        import os
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not qdrant_url or not qdrant_api_key:
            print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in environment variables", file=sys.stderr)
            sys.exit(1)

        if not neo4j_uri or not neo4j_user or not neo4j_password:
            print("❌ Error: NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME), and NEO4J_PASSWORD must be set in environment variables", file=sys.stderr)
            sys.exit(1)

        # Create reasoning engine
        engine = KGReasoningEngine(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            similarity_threshold=args.similarity_threshold,
        )

        # Execute query
        result = engine.query(args.query)

        # Close connections
        engine.close()

        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Result saved to: {args.output}")
        else:
            print("\n" + "="*80)
            print("ANSWER")
            print("="*80)
            print(result["answer"])
            print("="*80)

            if args.verbose:
                print("\nMETADATA:")
                print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def langgraph_command(args) -> None:
    """Run a query against the knowledge graph using LangGraph workflow."""
    print("🚀 Starting KG reasoning LangGraph workflow...")
    print(f"❓ Query: {args.query}")
    print(f"🤖 LLM Provider: {args.llm_provider}")
    print(f"🧠 LLM Model: {args.llm_model}")
    print(f"🎯 Similarity Threshold: {args.similarity_threshold}")
    print()

    try:
        # Check for environment variables
        import os
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not qdrant_url or not qdrant_api_key:
            print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in environment variables", file=sys.stderr)
            sys.exit(1)

        if not neo4j_uri or not neo4j_user or not neo4j_password:
            print("❌ Error: NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME), and NEO4J_PASSWORD must be set in environment variables", file=sys.stderr)
            sys.exit(1)

        # Execute LangGraph workflow
        result = run_langgraph_workflow(
            user_query=args.query,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            similarity_threshold=args.similarity_threshold,
        )

        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✅ Result saved to: {args.output}")
        else:
            print("\n" + "="*80)
            print("ANSWER")
            print("="*80)
            print(result["answer"])
            print("="*80)

            if args.verbose:
                print("\nMETADATA:")
                print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Reasoning Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s query "What is the relationship between Nvidia and AI?"
  %(prog)s query "How does debt affect company performance?" --verbose
  %(prog)s query "What are the main causes of market volatility?" --output result.json
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge graph")
    query_parser.add_argument(
        "query",
        type=str,
        help="Natural language query to process",
    )
    query_parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    query_parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model (default: gpt-4o-mini)",
    )
    query_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for entity matching (0.0-1.0, default: 0.75)",
    )
    query_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)",
    )
    query_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metadata",
    )

    # LangGraph query command
    langgraph_parser = subparsers.add_parser("langgraph", help="Query the knowledge graph using LangGraph workflow")
    langgraph_parser.add_argument(
        "query",
        type=str,
        help="Natural language query to process",
    )
    langgraph_parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    langgraph_parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model (default: gpt-4o-mini)",
    )
    langgraph_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for entity matching (0.0-1.0, default: 0.75)",
    )
    langgraph_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)",
    )
    langgraph_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed metadata",
    )

    args = parser.parse_args()

    try:
        # Handle query command
        if args.command == "query":
            query_command(args)
            return

        # Handle langgraph command
        if args.command == "langgraph":
            langgraph_command(args)
            return

        # No command specified, show help
        parser.print_help()
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
