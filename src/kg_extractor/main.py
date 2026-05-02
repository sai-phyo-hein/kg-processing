"""Main module for kg-extractor."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from kg_extractor import __version__
from kg_extractor.tools.agent import run_agent_interactive, run_agent_single_task
from kg_extractor.tools.langgraph_workflow import run_langgraph_workflow
from kg_extractor.utils.input_processor import DocumentProcessor
from kg_extractor.utils.markdown_formatter import (
    save_markdown_result,
    save_text_markdown,
)
from kg_extractor.utils.parser import (
    ImageEncodingError,
    NVIDIAAPIError,
    NVIDIAConfig,
    OpenRouterAPIError,
    OpenRouterConfig,
    OpenAIAPIError,
    OpenAIConfig,
    GoogleAPIError,
    GoogleConfig,
    extract_text_from_document,
    extract_text_from_document_openai,
    extract_text_from_document_openrouter,
    extract_text_from_document_google,
    extract_text_from_image,
    extract_text_from_image_openai,
    extract_text_from_image_openrouter,
    extract_text_from_image_google,
    extract_text_from_image_streaming,
    extract_text_from_image_streaming_openai,
    extract_text_from_image_streaming_openrouter,
    extract_text_from_image_streaming_google,
    get_api_key,
    get_openai_api_key,
    get_openrouter_api_key,
    get_google_api_key,
    process_document_with_api,
    process_document_with_openai,
    process_document_with_openrouter,
    process_document_with_google,
)
from kg_extractor.utils.triple_refiner import refine_triples_from_file
from kg_extractor.utils.neo4j_graph_builder import build_graph_from_file


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


def refine_command(args) -> None:
    """Refine knowledge graph triples using Qdrant for entity resolution."""
    print("🔍 Starting triple refinement with Qdrant...")
    print(f"📄 Input file: {args.input_file}")
    print(f"🤖 LLM Provider: {args.llm_provider}")
    print(f"🧠 LLM Model: {args.llm_model}")
    print()

    try:
        # Check for Qdrant configuration
        import os
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            print("❌ Error: QDRANT_URL and QDRANT_API_KEY must be set in environment variables", file=sys.stderr)
            sys.exit(1)

        # Refine triples
        output_path = refine_triples_from_file(
            input_path=args.input_file,
            output_path=args.output,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
        )

        # Read the output to get statistics
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        total_triples = output_data.get("total_triples", 0)
        total_chunks = output_data.get("total_chunks", 0)

        print("✅ Triple refinement completed successfully!")
        print(f"📊 Total triples refined: {total_triples}")
        print(f"🧩 Total chunks processed: {total_chunks}")
        print(f"📄 Output file: {output_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error reading triples file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error refining triples: {e}", file=sys.stderr)
        sys.exit(1)


def graph_command(args) -> None:
    """Build Neo4j graph from refined triples."""
    print("🕸️ Starting Neo4j graph building...")
    print(f"📄 Input file: {args.input_file}")
    print(f"📋 With schema: {args.with_schema}")
    print()

    try:
        # Check for Neo4j configuration
        import os
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not neo4j_uri or not neo4j_user or not neo4j_password:
            print("❌ Error: NEO4J_URI, NEO4J_USER (or NEO4J_USERNAME), and NEO4J_PASSWORD must be set in environment variables", file=sys.stderr)
            sys.exit(1)

        # Build graph
        stats = build_graph_from_file(
            input_path=args.input_file,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            with_schema=args.with_schema,
        )

        print("✅ Graph building completed successfully!")
        print(f"📊 Statistics:")
        print(f"   - Entities created: {stats['entities_created']}")
        print(f"   - Relationships created: {stats['relationships_created']}")
        if stats['errors']:
            print(f"   - Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"     - {error}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error building graph: {e}", file=sys.stderr)
        sys.exit(1)


def langgraph_command(args) -> None:
    """Run document processing using LangGraph workflow."""
    print("🚀 Starting LangGraph workflow...")
    print(f"📄 Input file: {args.input_file}")
    print(f"🤖 Provider: {args.provider}")
    print(f"🧠 Model: {args.model}")
    print(f"📝 Content type: {args.content_type}")
    print(f"🎯 Similarity threshold: {args.similarity_threshold}")
    print(f"🤖 Chunking LLM: {args.chunking_llm_provider}/{args.chunking_llm_model}")
    print(f"🔗 Triple LLM: {args.triplet_llm_provider}/{args.triplet_llm_model}")
    print(f"🔍 Refine triples: {args.refine_triples}")
    if args.refine_triples:
        print(f"🤖 Refinement LLM: {args.refinement_llm_provider}/{args.refinement_llm_model}")
    print(f"🕸️ Build graph: {args.build_graph}")
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
        provider=args.provider,
        model=args.model,
        content_type=args.content_type,
        similarity_threshold=args.similarity_threshold,
        output_format="markdown",
        chunking_llm_provider=args.chunking_llm_provider,
        chunking_llm_model=args.chunking_llm_model,
        triplet_llm_provider=args.triplet_llm_provider,
        triplet_llm_model=args.triplet_llm_model,
        refine_triples=args.refine_triples,
        refinement_llm_provider=args.refinement_llm_provider,
        refinement_llm_model=args.refinement_llm_model,
        build_graph=args.build_graph,
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
        description="Extract text from documents using NVIDIA API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg
  %(prog)s document.pdf
  %(prog)s --content-type table spreadsheet.xlsx
  %(prog)s --output result.json --format json presentation.pptx
  %(prog)s --max-tokens 4096 --temperature 0.3 report.docx
  %(prog)s --pages 1,2,4,6 document.pdf
  %(prog)s --pages 2-5 document.pdf
  %(prog)s langgraph document.pdf --similarity-threshold 0.5
  %(prog)s langgraph document.pdf --triplet-llm-provider openai --triplet-llm-model gpt-4o-mini
  %(prog)s langgraph document.pdf --until triple_extraction
  %(prog)s langgraph document.pdf --until semantic_chunking --no-refine-triples
  %(prog)s langgraph document.pdf --pages 1,3-5,7
  %(prog)s refine test_triples.json --llm-provider openai --llm-model gpt-4o-mini
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Default process command (backward compatibility)
    process_parser = subparsers.add_parser("process", help="Process a document file")
    process_parser.add_argument(
        "file_path",
        type=str,
        help="Path to document file (images, PDF, DOCX, PPTX, XLSX)",
    )
    process_parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming API response (images only)",
    )
    process_parser.add_argument(
        "--provider",
        type=str,
        choices=["nvidia", "openrouter", "openai", "groq", "google"],
        default="nvidia",
        help="API provider to use (default: nvidia)",
    )
    process_parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model to use (default: google/gemma-3-27b-it for nvidia, "
        "google/gemma-4-31b-it:free for openrouter, gpt-4o for openai, "
        "google/gemini-3.1-flash-lite-preview for google)",
    )
    process_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens in response (default: 2048)",
    )
    process_parser.add_argument(
        "--temperature",
        type=float,
        default=0.20,
        help="Sampling temperature (default: 0.20)",
    )
    process_parser.add_argument(
        "--top-p",
        type=float,
        default=0.70,
        help="Nucleus sampling parameter (default: 0.70)",
    )
    process_parser.add_argument(
        "--content-type",
        type=str,
        choices=["text", "diagram", "table", "mixed"],
        default="mixed",
        help="Type of content to focus on (default: mixed)",
    )
    process_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)",
    )
    process_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    process_parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Process only specific pages (e.g., '1,2,4,6' or '2-5' or '1,3-5,7')",
    )

    # LangGraph workflow command
    langgraph_parser = subparsers.add_parser("langgraph", help="Run document processing using LangGraph workflow")
    langgraph_parser.add_argument(
        "input_file",
        type=str,
        help="Input document file",
    )
    langgraph_parser.add_argument(
        "--provider",
        type=str,
        choices=["nvidia", "openrouter", "openai", "google"],
        default="nvidia",
        help="API provider (default: nvidia)",
    )
    langgraph_parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-4-multimodal-instruct",
        help="Model to use (default: microsoft/phi-4-multimodal-instruct for nvidia, "
        "gpt-4o for openai, google/gemini-3.1-flash-lite-preview for google)",
    )
    langgraph_parser.add_argument(
        "--content-type",
        type=str,
        choices=["text", "diagram", "table", "mixed"],
        default="mixed",
        help="Content type (default: mixed)",
    )
    langgraph_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for chunking (0.0-1.0, default: 0.5)",
    )
    langgraph_parser.add_argument(
        "--chunking-llm-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for chunking analysis (default: openai)",
    )
    langgraph_parser.add_argument(
        "--chunking-llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for chunking analysis (default: gpt-4o-mini)",
    )
    langgraph_parser.add_argument(
        "--triplet-llm-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for triple extraction (default: openai)",
    )
    langgraph_parser.add_argument(
        "--triplet-llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for triple extraction (default: gpt-4o-mini)",
    )
    langgraph_parser.add_argument(
        "--refine-triples",
        action="store_true",
        default=True,
        help="Refine triples using Qdrant for entity resolution (default: True)",
    )
    langgraph_parser.add_argument(
        "--no-refine-triples",
        dest="refine_triples",
        action="store_false",
        help="Skip triple refinement step",
    )
    langgraph_parser.add_argument(
        "--refinement-llm-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for triple refinement (default: openai)",
    )
    langgraph_parser.add_argument(
        "--refinement-llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for triple refinement (default: gpt-4o-mini)",
    )
    langgraph_parser.add_argument(
        "--build-graph",
        action="store_true",
        default=True,
        help="Build Neo4j graph from refined triples (default: True)",
    )
    langgraph_parser.add_argument(
        "--no-build-graph",
        dest="build_graph",
        action="store_false",
        help="Skip Neo4j graph building step",
    )
    langgraph_parser.add_argument(
        "--with-schema",
        action="store_true",
        default=False,
        help="Validate triples and graph against schema.md (default: False)",
    )
    langgraph_parser.add_argument(
        "--until",
        type=str,
        choices=["document_parsing", "metadata_extraction", "semantic_chunking", "triple_extraction", "triple_refining", "graph_building"],
        default=None,
        help="Stop workflow after this step (default: run all steps)",
    )
    langgraph_parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Process only specific pages (e.g., '1,2,4,6' or '2-5' or '1,3-5,7')",
    )

    # Graph command
    graph_parser = subparsers.add_parser("graph", help="Build Neo4j graph from refined triples")
    graph_parser.add_argument(
        "input_file",
        type=str,
        help="Input refined triples JSON file",
    )
    graph_parser.add_argument(
        "--with-schema",
        action="store_true",
        default=False,
        help="Validate triples and graph against schema.md (default: False)",
    )

    # Refine command
    refine_parser = subparsers.add_parser("refine", help="Refine knowledge graph triples using Qdrant for entity resolution")
    refine_parser.add_argument(
        "input_file",
        type=str,
        help="Input triples JSON file",
    )
    refine_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: input_path with _refined suffix)",
    )
    refine_parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "groq", "nvidia", "openrouter"],
        default="openai",
        help="LLM provider for canonical comparison (default: openai)",
    )
    refine_parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for canonical comparison (default: gpt-4o-mini)",
    )

    # For backward compatibility, also support the old argument format
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run in interactive agent mode with LangChain",
    )

    parser.add_argument(
        "--agent-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for agent (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--agent-task",
        type=str,
        help="Single task for agent mode (non-interactive)",
    )

    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        help="Path to document file (images, PDF, DOCX, PPTX, XLSX) - not required in agent mode",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming API response (images only)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        choices=["nvidia", "openrouter", "openai", "groq", "google"],
        default="nvidia",
        help="API provider to use (default: nvidia)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model to use (default: google/gemma-3-27b-it for nvidia, "
        "google/gemma-4-31b-it:free for openrouter, gpt-4o for openai, "
        "google/gemini-3.1-flash-lite-preview for google)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens in response (default: 2048)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.20,
        help="Sampling temperature (default: 0.20)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.70,
        help="Nucleus sampling parameter (default: 0.70)",
    )

    parser.add_argument(
        "--content-type",
        type=str,
        choices=["text", "diagram", "table", "mixed"],
        default="mixed",
        help="Type of content to focus on (default: mixed)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Process only specific pages (e.g., '1,2,4,6' or '2-5' or '1,3-5,7')",
    )

    args = parser.parse_args()

    try:
        # Handle langgraph command
        if args.command == "langgraph":
            langgraph_command(args)
            return

        # Handle refine command
        if args.command == "refine":
            refine_command(args)
            return

        # Handle graph command
        if args.command == "graph":
            graph_command(args)
            return

        # Handle process command
        if args.command == "process":
            # Map process command args to the old format
            args.file_path = args.file_path
            # Fall through to the processing logic below
        elif args.command is None and not args.agent and not args.file_path:
            # No command specified and no file path, show help
            parser.print_help()
            sys.exit(0)
        # Handle agent mode
        if args.agent:
            if args.agent_task:
                # Single task mode
                result = run_agent_single_task(
                    task=args.agent_task,
                    provider="openai",
                    model=args.agent_model,
                    debug=False,
                )
                print(result)
            else:
                # Interactive mode
                run_agent_interactive(
                    provider="openai",
                    model=args.agent_model,
                    debug=False,
                )
            return

        # Validate file path for non-agent mode
        if not args.file_path:
            print("Error: file_path is required when not in agent mode", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        file_path = Path(args.file_path)
        if not file_path.exists():
            print(f"Error: File not found: {args.file_path}", file=sys.stderr)
            sys.exit(1)

        if not file_path.is_file():
            print(f"Error: Path is not a file: {args.file_path}", file=sys.stderr)
            sys.exit(1)

        # Determine file type
        try:
            file_type = DocumentProcessor.get_file_type(args.file_path)
            if file_type == "unknown":
                print(
                    f"Error: Unsupported file format: {file_path.suffix}",
                    file=sys.stderr,
                )
                image_formats = ", ".join(sorted(DocumentProcessor.SUPPORTED_IMAGE_FORMATS))
                print(f"Supported formats: Images ({image_formats})", file=sys.stderr)
                document_formats = ", ".join(sorted(DocumentProcessor.SUPPORTED_DOCUMENT_FORMATS))
                print(f"Documents ({document_formats})", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to determine file type: {e}", file=sys.stderr)
            sys.exit(1)

        # Get API key and create configuration based on provider
        try:
            if args.provider == "nvidia":
                api_key = get_api_key()
                config = NVIDIAConfig(
                    api_key=api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=args.stream,
                )
            elif args.provider == "openai":
                api_key = get_openai_api_key()
                config = OpenAIConfig(
                    api_key=api_key,
                    model=args.model if args.model != "google/gemma-3-27b-it" else "gpt-4o",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=args.stream,
                )
            elif args.provider == "google":
                api_key = get_google_api_key()
                config = GoogleConfig(
                    api_key=api_key,
                    model=args.model if args.model != "google/gemma-3-27b-it" else "google/gemini-3.1-flash-lite-preview",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=args.stream,
                )
            else:  # openrouter
                api_key = get_openrouter_api_key()
                config = OpenRouterConfig(
                    api_key=api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=args.stream,
                )
        except (NVIDIAAPIError, OpenRouterAPIError, OpenAIAPIError, GoogleAPIError) as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.provider == "nvidia":
                print(
                    "\nPlease set your NVIDIA_API_KEY environment variable.",
                    file=sys.stderr,
                )
                print("You can get an API key from: https://build.nvidia.com/", file=sys.stderr)
            elif args.provider == "openai":
                print(
                    "\nPlease set your OPENAI_API_KEY environment variable.",
                    file=sys.stderr,
                )
                print("You can get an API key from: https://platform.openai.com/", file=sys.stderr)
            elif args.provider == "google":
                print(
                    "\nPlease set your GOOGLE_API_KEY environment variable.",
                    file=sys.stderr,
                )
                print("You can get an API key from: https://aistudio.google.com/app/apikey", file=sys.stderr)
            else:
                print(
                    "\nPlease set your OPENROUTER_API_KEY environment variable.",
                    file=sys.stderr,
                )
                print("You can get an API key from: https://openrouter.ai/", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid configuration: {e}", file=sys.stderr)
            sys.exit(1)

        # Parse pages argument if provided
        pages_to_process = None
        if hasattr(args, 'pages') and args.pages:
            try:
                pages_to_process = parse_pages_argument(args.pages)
                print(f"Processing pages: {', '.join(map(str, pages_to_process))}", file=sys.stderr)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        # Extract content based on file type
        try:
            if file_type == "image":
                # Handle images with streaming support
                if args.stream:
                    # Streaming mode
                    result = ""
                    if args.provider == "nvidia":
                        for chunk in extract_text_from_image_streaming(args.file_path, config):
                            if args.output:
                                result += chunk
                            else:
                                print(chunk, end="", flush=True)
                    elif args.provider == "openai":
                        for chunk in extract_text_from_image_streaming_openai(args.file_path, config):
                            if args.output:
                                result += chunk
                            else:
                                print(chunk, end="", flush=True)
                    elif args.provider == "google":
                        for chunk in extract_text_from_image_streaming_google(args.file_path, config):
                            if args.output:
                                result += chunk
                            else:
                                print(chunk, end="", flush=True)
                    else:  # openrouter
                        for chunk in extract_text_from_image_streaming_openrouter(
                            args.file_path, config
                        ):
                            if args.output:
                                result += chunk
                            else:
                                print(chunk, end="", flush=True)
                    if args.output:
                        with open(args.output, "w") as f:
                            f.write(result)
                        print(f"\nText saved to: {args.output}")
                    else:
                        print()  # Add newline after streaming
                else:
                    # Non-streaming mode
                    if args.provider == "nvidia":
                        text = extract_text_from_image(args.file_path, config)
                    elif args.provider == "openai":
                        text = extract_text_from_image_openai(args.file_path, config)
                    elif args.provider == "google":
                        text = extract_text_from_image_google(args.file_path, config)
                    else:  # openrouter
                        text = extract_text_from_image_openrouter(args.file_path, config)

                    if args.format == "markdown":
                        # Save as markdown
                        output_file = save_text_markdown(text, args.file_path)
                        print(f"Markdown saved to: {output_file}")
                    elif args.output:
                        with open(args.output, "w") as f:
                            f.write(text)
                        print(f"Text saved to: {args.output}")
                    else:
                        print(text)
            else:
                # Handle documents (PDF, DOCX, PPTX, XLSX)
                print(f"Processing {file_type.upper()} file...", file=sys.stderr)
                print(
                    f"Content type: {args.content_type}, Format: {args.format}",
                    file=sys.stderr,
                )

                if args.format == "json":
                    # Get structured JSON output
                    if args.provider == "nvidia":
                        result = extract_text_from_document(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    elif args.provider == "openai":
                        result = extract_text_from_document_openai(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    elif args.provider == "google":
                        result = extract_text_from_document_google(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    else:  # openrouter
                        result = extract_text_from_document_openrouter(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )

                    if args.output:
                        with open(args.output, "w") as f:
                            json.dump(result, f, indent=2)
                        print(f"JSON saved to: {args.output}")
                    else:
                        print(json.dumps(result, indent=2))
                elif args.format == "markdown":
                    # Get structured output and convert to markdown
                    if args.provider == "nvidia":
                        result = process_document_with_api(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    elif args.provider == "openai":
                        result = process_document_with_openai(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    elif args.provider == "google":
                        result = process_document_with_google(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    else:  # openrouter
                        result = process_document_with_openrouter(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )

                    # Save as markdown
                    output_file = save_markdown_result(result, args.file_path)
                    print(f"Markdown saved to: {output_file}")
                else:
                    # Get text output
                    if args.provider == "nvidia":
                        text = extract_text_from_document(args.file_path, config, args.content_type, pages=pages_to_process)
                    elif args.provider == "openai":
                        text = extract_text_from_document_openai(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    elif args.provider == "google":
                        text = extract_text_from_document_google(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )
                    else:  # openrouter
                        text = extract_text_from_document_openrouter(
                            args.file_path, config, args.content_type, pages=pages_to_process
                        )

                    if args.output:
                        with open(args.output, "w") as f:
                            f.write(text)
                        print(f"Text saved to: {args.output}")
                    else:
                        print(text)

        except ImageEncodingError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except (NVIDIAAPIError, OpenRouterAPIError, OpenAIAPIError, GoogleAPIError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
