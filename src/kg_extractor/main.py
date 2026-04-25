"""Main module for kg-extractor."""

import argparse
import json
import sys
from pathlib import Path

from kg_extractor import __version__
from kg_extractor.agent import run_agent_interactive, run_agent_single_task
from kg_extractor.input_processor import DocumentProcessor
from kg_extractor.markdown_formatter import (
    save_markdown_result,
    save_text_markdown,
)
from kg_extractor.parser import (
    ImageEncodingError,
    NVIDIAAPIError,
    NVIDIAConfig,
    OpenRouterAPIError,
    OpenRouterConfig,
    extract_text_from_document,
    extract_text_from_document_openrouter,
    extract_text_from_image,
    extract_text_from_image_openrouter,
    extract_text_from_image_streaming,
    extract_text_from_image_streaming_openrouter,
    get_api_key,
    get_openrouter_api_key,
    process_document_with_api,
    process_document_with_openrouter,
)


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
        """,
    )

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
        choices=["nvidia", "openrouter", "groq"],
        default="nvidia",
        help="API provider to use (default: nvidia)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model to use (default: google/gemma-3-27b-it for nvidia, "
        "google/gemma-4-31b-it:free for openrouter)",
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

    args = parser.parse_args()

    try:
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
        except (NVIDIAAPIError, OpenRouterAPIError) as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.provider == "nvidia":
                print(
                    "\nPlease set your NVIDIA_API_KEY environment variable.",
                    file=sys.stderr,
                )
                print("You can get an API key from: https://build.nvidia.com/", file=sys.stderr)
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
                    else:
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
                    else:
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
                            args.file_path, config, args.content_type
                        )
                    else:
                        result = extract_text_from_document_openrouter(
                            args.file_path, config, args.content_type
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
                            args.file_path, config, args.content_type
                        )
                    else:
                        result = process_document_with_openrouter(
                            args.file_path, config, args.content_type
                        )

                    # Save as markdown
                    output_file = save_markdown_result(result, args.file_path)
                    print(f"Markdown saved to: {output_file}")
                else:
                    # Get text output
                    if args.provider == "nvidia":
                        text = extract_text_from_document(args.file_path, config, args.content_type)
                    else:
                        text = extract_text_from_document_openrouter(
                            args.file_path, config, args.content_type
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
        except (NVIDIAAPIError, OpenRouterAPIError) as e:
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
