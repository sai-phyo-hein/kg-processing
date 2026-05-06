"""Main module for kg-reasoning."""

import argparse
import json
import sys

from kg_extractor.utils.model_setup import (
    REASONING_PROVIDER,
    ORCHESTRATOR_MODEL,
    WORKER_MODEL,
    SYNTHESIZER_MODEL,
)
from kg_reasoning.workflows.multi_agent_workflow import run_multi_agent_workflow


def query_command(args) -> None:
    """Run a query using the multi-agent workflow."""
    print("🚀 Starting Multi-Agent KG Reasoning Workflow...")
    print(f"❓ Query: {args.query}")
    print(f"🤖 LLM Provider: {REASONING_PROVIDER}")
    print(f"🎯 Orchestrator Model: {ORCHESTRATOR_MODEL}")
    print(f"⚙️  Worker Model: {WORKER_MODEL}")
    print(f"📝 Synthesizer Model: {SYNTHESIZER_MODEL}")
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

        # Execute multi-agent workflow
        result = run_multi_agent_workflow(
            user_query=args.query,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )

        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Result saved to: {args.output}")
        else:
            print("\n" + "="*80)
            print("FINAL ANSWER")
            print("="*80)
            print(result.get("final_answer", "No answer generated"))
            print("="*80)

            if args.verbose:
                print("\nWORKFLOW DETAILS:")
                print(f"  - Strategies executed: {len(result.get('strategies', []))}")
                print(f"  - Entities found: {len(result.get('entities_found', []))}")
                print(f"  - Predicates found: {len(result.get('predicates_found', []))}")
                print(f"  - Total results: {result.get('total_results_count', 0)}")
                print(f"  - Markdown files: {len(result.get('markdown_files', []))}")
                print(f"  - Execution time: {result.get('execution_time_seconds', 0):.2f}s")
                print(f"  - Answer quality: {result.get('synthesis_quality', 'unknown')}")

                if result.get("markdown_files"):
                    print(f"\n  Result files:")
                    for file in result["markdown_files"]:
                        print(f"    - {file}")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Reasoning Engine (Multi-Agent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s query "What drives Thailand ecommerce?"
  %(prog)s query "How does debt affect company performance?" --verbose
  %(prog)s query "What are the main causes of market volatility?" --output result.json

Model configuration is done via environment variables (see .env):
  REASONING_PROVIDER, ORCHESTRATOR_MODEL, WORKER_MODEL, SYNTHESIZER_MODEL
  OPENAI_EMBEDDING_MODEL
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    query_parser = subparsers.add_parser(
        "query",
        help="Query the knowledge graph using the multi-agent workflow (Orchestrator → Workers → Synthesizer)",
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="Natural language query to process",
    )
    query_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)",
    )
    query_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed workflow metadata",
    )

    args = parser.parse_args()

    try:
        if args.command == "query":
            query_command(args)
            return

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
