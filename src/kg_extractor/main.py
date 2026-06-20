"""Main module for kg-extractor."""

import argparse
import sys
from typing import List, Optional

from kg_extractor import __version__
from kg_extractor.utils.model_setup import (
    PARSING_PROVIDER,
    PARSING_MODEL,
    CHUNKING_PROVIDER,
    CHUNKING_MODEL,
    TRIPLET_PROVIDER,
    TRIPLET_MODEL,
    REFINEMENT_PROVIDER,
    REFINEMENT_MODEL,
)
from kg_extractor.workflow.langgraph_workflow import (
    run_langgraph_workflow,
    run_parse_document_only,
    run_extract_metadata_only,
    run_chunk_document_only,
    run_extract_triples_only,
    run_translate_triples_only,
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

    # --- Single-node debug mode: --parse-document ---
    if args.parse_document:
        print("📄 [Debug] Running ONLY parse-document node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"🤖 Parsing provider: {PARSING_PROVIDER}/{PARSING_MODEL}")
        if args.pages:
            try:
                pages_to_process = parse_pages_argument(args.pages)
                print(f"📄 Processing pages: {', '.join(map(str, pages_to_process))}")
            except ValueError as e:
                print(f"❌ Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            pages_to_process = None
        print()
        result = run_parse_document_only(
            input_file=args.input_file,
            pages=pages_to_process,
            location_moo=args.location_moo,
            location_village=args.location_village,
        )
        if result["status"] == "success":
            print("✅ Parse-document node completed successfully!")
            print(f"📄 Markdown output: {result['markdown_output']}")
        else:
            print(f"❌ Parse-document node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-node debug mode: --extract-metadata ---
    if args.extract_metadata:
        print("📊 [Debug] Running ONLY extract-metadata node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"🤖 Chunking LLM: {CHUNKING_PROVIDER}/{CHUNKING_MODEL}")
        if args.location_moo:
            print(f"📍 Location moo: {args.location_moo}")
        if args.location_village:
            print(f"📍 Location village: {args.location_village}")
        print()
        result = run_extract_metadata_only(
            input_file=args.input_file,
            location_moo=args.location_moo,
            location_village=args.location_village,
        )
        if result["status"] == "success":
            print("✅ Extract-metadata node completed successfully!")
            if result["metadata"]:
                print(f"📊 Metadata extracted: {result['metadata'].get('unique_id')}")
        else:
            print(f"❌ Extract-metadata node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-node debug mode: --chunk-document ---
    if args.chunk_document:
        print("🧩 [Debug] Running ONLY chunk-document node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"🤖 Chunking LLM: {CHUNKING_PROVIDER}/{CHUNKING_MODEL}")
        print(f"🎯 Chunk granularity: {args.chunk_granularity}")
        print()
        result = run_chunk_document_only(
            input_file=args.input_file,
            chunk_granularity=args.chunk_granularity,
            location_moo=args.location_moo,
            location_village=args.location_village,
        )
        if result["status"] == "success":
            print("✅ Chunk-document node completed successfully!")
            print(f"🧩 Chunks output: {result['chunks_output']}")
        else:
            print(f"❌ Chunk-document node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-node debug mode: --extract-triples ---
    if args.extract_triples:
        print("🔗 [Debug] Running ONLY extract-triples node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"🤖 Triple LLM: {TRIPLET_PROVIDER}/{TRIPLET_MODEL}")
        if args.chunk_id is not None:
            print(f"🎯 Only chunk_id: {args.chunk_id} (other chunks preserved)")
        print()
        result = run_extract_triples_only(
            input_file=args.input_file,
            location_moo=args.location_moo,
            location_village=args.location_village,
            chunk_id=args.chunk_id,
        )
        if result["status"] == "success":
            print("✅ Extract-triples node completed successfully!")
            print(f"🔗 Triples output: {result['triples_output']}")
        else:
            print(f"❌ Extract-triples node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-node debug mode: --translate-triples ---
    if args.translate_triples:
        print("🌐 [Debug] Running ONLY translate-triples node...")
        print(f"📄 Input file: {args.input_file}")
        if args.chunk_id is not None:
            print(f"🎯 Only chunk_id: {args.chunk_id} (other chunks preserved)")
        print()
        result = run_translate_triples_only(
            input_file=args.input_file,
            location_moo=args.location_moo,
            location_village=args.location_village,
            chunk_id=args.chunk_id,
        )
        if result["status"] == "success":
            print("✅ Translate-triples node completed successfully!")
            print(f"🌐 Triples output: {result['triples_output']}")
        else:
            print(f"❌ Translate-triples node failed: {result['error']}", file=sys.stderr)
            sys.exit(1)
        return

    # --- Single-node debug mode: --refine-triples ---
    if args.refine_triples:
        print("🔍 [Debug] Running ONLY refine-triples node...")
        print(f"📄 Input file: {args.input_file}")
        print(f"🤖 Refinement LLM: {REFINEMENT_PROVIDER}/{REFINEMENT_MODEL}")
        print(f"🔍 Similarity threshold: {args.similarity_threshold}")
        print()
        result = run_refine_triples_only(
            input_file=args.input_file,
            similarity_threshold=args.similarity_threshold,
            location_moo=args.location_moo,
            location_village=args.location_village,
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
        print()
        result = run_build_graph_only(
            input_file=args.input_file,
            location_moo=args.location_moo,
            location_village=args.location_village,
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
    print(f"🤖 Parsing provider: {PARSING_PROVIDER}")
    print(f"🧠 Parsing model: {PARSING_MODEL}")
    print(f"🎯 Chunk granularity: {args.chunk_granularity}")
    print(f"🔍 Similarity threshold: {args.similarity_threshold}")
    print(f"🤖 Chunking LLM: {CHUNKING_PROVIDER}/{CHUNKING_MODEL}")
    print(f"🔗 Triple LLM: {TRIPLET_PROVIDER}/{TRIPLET_MODEL}")
    print(f"🤖 Refinement LLM: {REFINEMENT_PROVIDER}/{REFINEMENT_MODEL}")
    if args.location_moo:
        print(f"📍 Location moo: {args.location_moo}")
    if args.location_village:
        print(f"📍 Location village: {args.location_village}")
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
        chunk_granularity=args.chunk_granularity,
        similarity_threshold=args.similarity_threshold,
        until_step=args.until,
        pages=pages_to_process,
        location_moo=args.location_moo,
        location_village=args.location_village,
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
  %(prog)s document.pdf --similarity-threshold 0.95
  %(prog)s document.pdf --until triple_extraction
  %(prog)s document.pdf --pages 1,3-5,7

Model configuration is done via environment variables (see .env):
  PARSING_PROVIDER, PARSING_MODEL
  CHUNKING_PROVIDER, CHUNKING_MODEL
  TRIPLET_PROVIDER, TRIPLET_MODEL
  REFINEMENT_PROVIDER, REFINEMENT_MODEL
  OPENAI_EMBEDDING_MODEL
        """,
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Input document file",
    )
    parser.add_argument(
        "--chunk-granularity",
        type=float,
        default=0.1,
        help="Granularity for semantic chunking (0.0=very fine, 1.0=coarse, default: 0.2)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for entity resolution in triple refiner (0.0-1.0, default: 0.95)",
    )
    parser.add_argument(
        "--parse-document",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the parse-document node on input_file, then stop. "
        "Without this flag the full pipeline runs (parsing included).",
    )
    parser.add_argument(
        "--extract-metadata",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the extract-metadata node using the existing markdown file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (metadata included).",
    )
    parser.add_argument(
        "--chunk-document",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the chunk-document node using the existing markdown file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (chunking included).",
    )
    parser.add_argument(
        "--extract-triples",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the extract-triples node using the existing chunks file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (extraction included).",
    )
    parser.add_argument(
        "--translate-triples",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the translate-triples node using the existing triples file "
        "derived from input_file, translating Thai/mixed chunks in place, then stop. "
        "Without this flag the full pipeline runs (translation included).",
    )
    parser.add_argument(
        "--refine-triples",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the refine-triples node using the existing triples file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (refine included).",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        default=False,
        help="Debug mode: run ONLY the build-graph node using the existing refined triples file "
        "derived from input_file, then stop. Without this flag the full pipeline runs (build included).",
    )
    parser.add_argument(
        "--until",
        type=str,
        choices=["document_parsing", "metadata_extraction", "semantic_chunking", "triple_extraction", "triple_translation", "triple_refining", "graph_building"],
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
        "--chunk-id",
        "--chunk_id",
        type=int,
        default=None,
        help="With --extract-triples or --translate-triples: (re-)process only "
        "this chunk_id and merge the result into the existing triples file in "
        "place (other chunks are preserved). Ignored without those flags.",
    )
    parser.add_argument(
        "--location-moo",
        type=str,
        default=None,
        help="Location moo value (e.g., 'หมู่ 1') to use instead of extracting from document",
    )
    parser.add_argument(
        "--location-village",
        type=str,
        default=None,
        help="Location village value (e.g., 'บ้านสวน') to use instead of extracting from document",
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
