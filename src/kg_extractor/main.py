"""Main module for kg-extractor."""

import argparse
import sys
from typing import List, Optional

from kg_extractor import __version__
from kg_extractor.workflow.langgraph_workflow import (
    run_langgraph_workflow,
    run_refine_triples_only,
    run_build_graph_only,
)


def parse_pages_argument(pages_str: str) -> Optional[List[int]]:
    """Parse pages argument string into list of page numbers.

    Args:
        pages_str: String like "1,2,4,6" or "2-5" or "1,3-5,7"

    Returns:
        List of page numbers (1-indexed), or None if pages_str is None/empty

    Raises:
        ValueError: If pages string format is invalid
    """
    if not pages_str:
        return None

    pages = set()
    parts = pages_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like "2-5"
            try:
                start, end = part.split('-')
                start_num = int(start.strip())
                end_num = int(end.strip())
                if start_num < 1 or end_num < 1:
                    raise ValueError("Page numbers must be positive integers")
                if start_num > end_num:
                    raise ValueError(f"Invalid range: {part}. Start must be <= end")
                pages.update(range(start_num, end_num + 1))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid page range format: {part}")
                raise
        else:
            # Handle single page like "4"
            try:
                page_num = int(part)
                if page_num < 1:
                    raise ValueError("Page numbers must be positive integers")
                pages.add(page_num)
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")

    return sorted(list(pages))


def langgraph_command(args) -> None:
    """Run document processing using LangGraph workflow."""

    # --- Single-node debug mode: --refine-triples ---
    if args.refine_triples:
        print("🔍 [Debug] Running ONLY refine-triples node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"🤖 Refinement LLM: {args.refinement_provider}/{args.refinement_model}")
        print(f"🔍 Similarity threshold: {args.similarity_threshold}")
        print()
        result = run_refine_triples_only(
            input_file=args.input_file,
            refinement_llm_provider=args.refinement_provider,
            refinement_llm_model=args.refinement_model,
            similarity_threshold=args.similarity_threshold,
        )
        if result["status"] == "success":
            print("✅ Refine-triples node completed successfully!")
            print(f"🔍 Refined triples: {result['refined_output']}")
        else:
            print(f"❌ Refine-triples node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-node debug mode: --build-graph ---
    if args.build_graph:
        print("🕸️ [Debug] Running ONLY build-graph node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"📋 With schema: {args.with_schema}")
        print()
        result = run_build_graph_only(
            input_file=args.input_file,
            with_schema=args.with_schema,
        )
        if result["status"] == "success":
            print("✅ Build-graph node completed successfully!")
            if result['graph_stats']:
                stats = result['graph_stats']
                print(f"🕸️ Graph built:")
                print(f"   - Entities: {stats['entities_created']}")
                print(f"   - Relationships: {stats['relationships_created']}")
                if stats['errors']:
                    print(f"   - Errors: {len(stats['errors'])}")
        else:
            print(f"❌ Build-graph node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Full pipeline ---
    print("🚀 Starting LangGraph workflow...")
    print(f"📄 Input file: {args.input_file}")
    print(f"🤖 Parsing provider: {args.parsing_provider}")
    print(f"🧠 Parsing model: {args.parsing_model}")
    print(f"🎯 Chunk granularity: {args.chunk_granularity}")
    print(f"🔍 Similarity threshold: {args.similarity_threshold}")
    print(f"🤖 Chunking LLM: {args.chunking_provider}/{args.chunking_model}")
    print(f"🔗 Triple LLM: {args.triplet_provider}/{args.triplet_model}")
    print(f"🤖 Refinement LLM: {args.refinement_provider}/{args.refinement_model}")
    print(f"📋 With schema: {args.with_schema}")
    if args.until:
        print(f"⏹️ Stop after: {args.until}")

    # Parse pages argument if provided
    pages_to_process = None
    if args.pages:
        try:
            pages_to_process = parse_pages_argument(args.pages)
            print(f"📄 Processing pages: {', '.join(map(str, pages_to_process))}")
        except ValueError as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)

    print()

    result = run_langgraph_workflow(
        input_file=args.input_file,
        provider=args.parsing_provider,
        model=args.parsing_model,
        chunk_granularity=args.chunk_granularity,
        similarity_threshold=args.similarity_threshold,
        chunking_llm_provider=args.chunking_provider,
        chunking_llm_model=args.chunking_model,
        triplet_llm_provider=args.triplet_provider,
        triplet_llm_model=args.triplet_model,
        refinement_llm_provider=args.refinement_provider,
        refinement_llm_model=args.refinement_model,
        with_schema=args.with_schema,
        until_step=args.until,
        pages=pages_to_process,
    )

    if result["status"] == "success":
        print("✅ LangGraph workflow completed successfully!")
        print(f"📄 Markdown output: {result['markdown_output']}")
        if result['metadata']:
            print(f"📊 Metadata extracted: {result['metadata'].get('unique_id')}")
        print(f"🧩 Chunks output: {result['chunks_output']}")
        print(f"🔗 Triples output: {result['triples_output']}")
        if result['refined_output']:
            print(f"🔍 Refined triples: {result['refined_output']}")
        if result['graph_stats']:
            stats = result['graph_stats']
            print(f"🕸️ Graph built:")
            print(f"   - Entities: {stats['entities_created']}")
            print(f"   - Relationships: {stats['relationships_created']}")
            if stats['errors']:
                print(f"   - Errors: {len(stats['errors'])}")
    else:
        print(f"❌ LangGraph workflow failed: {result['error']}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Extractor - Process documents and build knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf --chunk-granularity 0.5
  %(prog)s document.pdf --similarity-threshold 0.85
  %(prog)s document.pdf --triplet-provider openai --triplet-model gpt-4o-mini
  %(prog)s document.pdf --until triple_extraction
  %(prog)s document.pdf --pages 1,3-5,7
        """,
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Input document file",
    )
    parser.add_argument(
        "--parsing-provider",
        type=str,
        choices=["nvidia", "openrouter", "openai", "google"],
        default="nvidia",
        help="API provider for document parsing (default: nvidia)",
    )
    parser.add_argument(
        "--parsing-model",
        type=str,
        default="microsoft/phi-4-multimodal-instruct",
        help="Model for document parsing (default: microsoft/phi-4-multimodal-instruct for nvidia, "
        "gpt-4o for openai, google/gemini-3.1-flash-lite-preview for google)",
    )
    parser.add_argument(
        "--chunk-granularity",
        type=float,
        default=0.5,
        help="Granularity for semantic chunking (0.0=very fine, 1.0=coarse, default: 0.5)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for entity resolution in triple refiner (0.0-1.0, default: 0.85)",
    )
    parser.add_argument(
        "--chunking-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for chunking analysis (default: openai)",
    )
    parser.add_argument(
        "--chunking-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for chunking analysis (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--triplet-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for triple extraction (default: openai)",
    )
    parser.add_argument(
        "--triplet-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for triple extraction (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--refine-triples",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the refine-triples node using the existing triples file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (refine included).",
    )
    parser.add_argument(
        "--refinement-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for triple refinement (default: openai)",
    )
    parser.add_argument(
        "--refinement-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for triple refinement (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the build-graph node using the existing refined triples file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (build included).",
    )
    parser.add_argument(
        "--with-schema",
        action="store_true",
        default=False,
        help="Validate triples and graph against schema.md (default: False)",
    )
    parser.add_argument(
        "--until",
        type=str,
        choices=["document_parsing", "metadata_extraction", "semantic_chunking", "triple_extraction", "triple_refining", "graph_building"],
        default=None,
        help="Stop workflow after this step (default: run all steps)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Process only specific pages (e.g., '1,2,4,6' or '2-5' or '1,3-5,7')",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    try:
        langgraph_command(args)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
