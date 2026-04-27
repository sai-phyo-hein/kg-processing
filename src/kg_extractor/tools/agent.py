"""LangChain agent for document processing and markdown editing."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

from kg_extractor.utils.input_processor import DocumentProcessor
from kg_extractor.utils.markdown_formatter import save_markdown_result, save_text_markdown
from kg_extractor.utils.parser import (
    GroqAPIError,
    NVIDIAAPIError,
    NVIDIAConfig,
    OpenRouterAPIError,
    OpenRouterConfig,
    extract_text_from_document,
    extract_text_from_document_openrouter,
    get_api_key,
    get_groq_api_key,
    get_openrouter_api_key,
    process_document_with_api,
    process_document_with_openrouter,
)
from kg_extractor.utils.semantic_chunker import chunk_markdown_file
from kg_extractor.utils.triple_extractor import extract_triples_from_chunks
from kg_extractor.utils.triple_refiner import refine_triples_from_file


@tool
def process_document_tool(
    file_path: str,
    provider: str = "nvidia",
    model: str = "microsoft/phi-4-multimodal-instruct",
    content_type: str = "mixed",
    output_format: str = "markdown",
) -> str:
    """Process a document file and extract structured content.

    Args:
        file_path: Path to the document file (images, PDF, DOCX, PPTX, XLSX)
        provider: API provider to use (nvidia or openrouter)
        model: Model to use for processing
        content_type: Type of content to focus on (text, diagram, table, mixed)
        output_format: Output format (text, json, markdown)

    Returns:
        Result of document processing with status and output file path
    """
    try:
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        # Get file type
        file_type = DocumentProcessor.get_file_type(file_path)
        if file_type == "unknown":
            return f"Error: Unsupported file format: {path.suffix}"

        # Get API key and create configuration
        if provider == "nvidia":
            api_key = get_api_key()
            config = NVIDIAConfig(
                api_key=api_key,
                model=model,
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        else:  # openrouter
            api_key = get_openrouter_api_key()
            config = OpenRouterConfig(
                api_key=api_key,
                model=model,
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )

        # Process document based on output format
        if output_format == "json":
            if provider == "nvidia":
                result = extract_text_from_document(file_path, config, content_type)
            else:
                result = extract_text_from_document_openrouter(file_path, config, content_type)

            # Save JSON result
            output_file = Path("output") / f"{path.stem}_analysis.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            return f"Successfully processed {file_type.upper()} file. JSON saved to: {output_file}"

        elif output_format == "markdown":
            if provider == "nvidia":
                result = process_document_with_api(file_path, config, content_type)
            else:
                result = process_document_with_openrouter(file_path, config, content_type)

            # Save markdown result
            output_file = save_markdown_result(result, file_path)
            return f"Successfully processed {file_type.upper()} file. Markdown saved to: {output_file}"

        else:  # text format
            if provider == "nvidia":
                text = extract_text_from_document(file_path, config, content_type)
            else:
                text = extract_text_from_document_openrouter(file_path, config, content_type)

            # Save text result
            output_file = save_text_markdown(text, file_path)
            return f"Successfully processed {file_type.upper()} file. Text saved to: {output_file}"

    except (NVIDIAAPIError, OpenRouterAPIError) as e:
        return f"API Error: {e}"
    except Exception as e:
        return f"Error processing document: {e}"


@tool
def read_markdown_file(file_path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    """Read a markdown file line by line.

    Args:
        file_path: Path to the markdown file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (optional, reads to end if not specified)

    Returns:
        Content of the specified lines from the markdown file
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Adjust for 1-indexed line numbers
        start_idx = max(0, start_line - 1)
        end_idx = len(lines) if end_line is None else min(len(lines), end_line)

        if start_idx >= len(lines):
            return f"Error: Start line {start_line} is beyond file length ({len(lines)} lines)"

        selected_lines = lines[start_idx:end_idx]
        content = "".join(selected_lines)

        return f"Lines {start_line}-{end_idx if end_line else len(lines)} of {file_path}:\n{content}"

    except Exception as e:
        return f"Error reading file: {e}"


@tool
def edit_markdown_file(
    file_path: str,
    line_number: int,
    new_content: str,
    mode: str = "replace",
) -> str:
    """Edit a markdown file at a specific line.

    Args:
        file_path: Path to the markdown file
        line_number: Line number to edit (1-indexed)
        new_content: New content to insert/replace
        mode: Edit mode - 'replace' (replace the line), 'insert' (insert before line), 'append' (insert after line)

    Returns:
        Result of the edit operation
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Adjust for 1-indexed line number
        line_idx = line_number - 1

        if line_idx < 0 or line_idx >= len(lines):
            return f"Error: Line number {line_number} is out of range (file has {len(lines)} lines)"

        if mode == "replace":
            # Ensure new content ends with newline if replacing a line
            if not new_content.endswith("\n"):
                new_content += "\n"
            lines[line_idx] = new_content
            action = f"replaced line {line_number}"

        elif mode == "insert":
            # Insert new content before the specified line
            if not new_content.endswith("\n"):
                new_content += "\n"
            lines.insert(line_idx, new_content)
            action = f"inserted content before line {line_number}"

        elif mode == "append":
            # Insert new content after the specified line
            if not new_content.endswith("\n"):
                new_content += "\n"
            lines.insert(line_idx + 1, new_content)
            action = f"appended content after line {line_number}"

        else:
            return f"Error: Invalid mode '{mode}'. Use 'replace', 'insert', or 'append'"

        # Write back to file
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        return f"Successfully {action} in {file_path}"

    except Exception as e:
        return f"Error editing file: {e}"


@tool
def search_markdown_content(file_path: str, search_term: str, case_sensitive: bool = False) -> str:
    """Search for content in a markdown file.

    Args:
        file_path: Path to the markdown file
        search_term: Term to search for
        case_sensitive: Whether search should be case sensitive

    Returns:
        Lines containing the search term with line numbers
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        search_term_cmp = search_term if case_sensitive else search_term.lower()
        results = []

        for i, line in enumerate(lines, 1):
            line_cmp = line if case_sensitive else line.lower()
            if search_term_cmp in line_cmp:
                results.append(f"Line {i}: {line.rstrip()}")

        if not results:
            return f"No matches found for '{search_term}' in {file_path}"

        return f"Found {len(results)} matches for '{search_term}' in {file_path}:\n" + "\n".join(
            results
        )

    except Exception as e:
        return f"Error searching file: {e}"


@tool
def list_markdown_files(directory: str = "output") -> str:
    """List all markdown files in a directory.

    Args:
        directory: Directory to search for markdown files

    Returns:
        List of markdown files with their sizes and line counts
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"

        md_files = list(dir_path.glob("*.md"))

        if not md_files:
            return f"No markdown files found in {directory}"

        results = []
        for md_file in sorted(md_files):
            size = md_file.stat().st_size
            with open(md_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            results.append(f"- {md_file.name} ({size} bytes, {line_count} lines)")

        return f"Found {len(md_files)} markdown file(s) in {directory}:\n" + "\n".join(results)

    except Exception as e:
        return f"Error listing files: {e}"


@tool
def semantic_chunk_markdown(
    file_path: str,
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> str:
    """Perform semantic chunking on a markdown file using LLM analysis.

    The LLM reads the content and determines where topic changes occur based on its understanding.
    Output chunks contain only chunk_id and content for simplicity.

    Args:
        file_path: Path to the markdown file to chunk
        similarity_threshold: Threshold for detecting topic changes (0.0-1.0)
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
        llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
        llm_model: Model to use for LLM analysis

    Returns:
        Result of chunking with chunk count and output file path
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        # Validate parameters
        if not 0.0 <= similarity_threshold <= 1.0:
            return f"Error: similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
        if min_chunk_size < 0:
            return f"Error: min_chunk_size must be non-negative, got {min_chunk_size}"
        if max_chunk_size < min_chunk_size:
            return f"Error: max_chunk_size ({max_chunk_size}) must be >= min_chunk_size ({min_chunk_size})"

        # Perform chunking
        output_path = chunk_markdown_file(
            file_path=file_path,
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            llm_provider=llm_provider,
            llm_model=llm_model,
            output_dir=str(Path(__file__).parent.parent.parent.parent / "output"),
        )

        # Read the output to get chunk count
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        chunk_count = output_data.get("total_chunks", 0)

        return f"Successfully chunked {file_path} into {chunk_count} chunks using {llm_provider}/{llm_model}. Output saved to: {output_path}"

    except FileNotFoundError as e:
        return f"Error: {e}"
    except json.JSONDecodeError as e:
        return f"Error reading chunk output: {e}"
    except Exception as e:
        return f"Error chunking markdown: {e}"


@tool
def extract_triples_tool(
    chunks_file: str,
    source_file: str = "",
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> str:
    """Extract knowledge graph triples from chunks using LLM analysis.

    The LLM analyzes each chunk and extracts triples with two-level predicate hierarchy
    and temporal metadata for knowledge graph construction.

    Args:
        chunks_file: Path to the chunks JSON file
        source_file: Original source file path for context
        llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
        llm_model: Model to use for LLM analysis

    Returns:
        Result of triple extraction with triple count and output file path
    """
    try:
        path = Path(chunks_file)
        if not path.exists():
            return f"Error: Chunks file not found: {chunks_file}"

        # Read chunks
        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = chunks_data.get("chunks", [])

        if not chunks:
            return f"Error: No chunks found in {chunks_file}"

        # Extract triples
        output_path = extract_triples_from_chunks(
            chunks=chunks,
            source_file=source_file,
            llm_provider=llm_provider,
            llm_model=llm_model,
            output_dir=str(Path(__file__).parent.parent.parent.parent / "output"),
        )

        # Read the output to get triple count
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        triple_count = output_data.get("total_triples", 0)

        return f"Successfully extracted {triple_count} triples from {len(chunks)} chunks using {llm_provider}/{llm_model}. Output saved to: {output_path}"

    except FileNotFoundError as e:
        return f"Error: {e}"
    except json.JSONDecodeError as e:
        return f"Error reading chunks file: {e}"
    except Exception as e:
        return f"Error extracting triples: {e}"


@tool
def refine_triples_tool(
    triples_file: str,
    output_path: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> str:
    """Refine knowledge graph triples using Qdrant for entity resolution.

    This tool queries Qdrant for existing entities, predicates, and ontologies,
    uses LLM to determine canonical forms, and updates triples with canonical references.

    Args:
        triples_file: Path to the triples JSON file to refine
        output_path: Path to save refined triples (default: input_path with _refined suffix)
        llm_provider: LLM provider for canonical comparison (openai, groq, nvidia, openrouter)
        llm_model: Model for LLM analysis

    Returns:
        Result of triple refinement with output file path
    """
    try:
        path = Path(triples_file)
        if not path.exists():
            return f"Error: Triples file not found: {triples_file}"

        # Check for Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            return f"Error: QDRANT_URL and QDRANT_API_KEY must be set in environment variables"

        # Refine triples
        output_path_result = refine_triples_from_file(
            input_path=triples_file,
            output_path=output_path,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        # Read the output to get statistics
        with open(output_path_result, "r", encoding="utf-8") as f:
            output_data = json.load(f)
        total_triples = output_data.get("total_triples", 0)
        total_chunks = output_data.get("total_chunks", 0)

        return f"Successfully refined {total_triples} triples from {total_chunks} chunks using {llm_provider}/{llm_model} and Qdrant. Output saved to: {output_path_result}"

    except FileNotFoundError as e:
        return f"Error: {e}"
    except json.JSONDecodeError as e:
        return f"Error reading triples file: {e}"
    except Exception as e:
        return f"Error refining triples: {e}"


def create_agent(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    debug: bool = False,
) -> Any:
    """Create a LangChain agent with document processing and markdown editing tools.

    Args:
        provider: API provider for the LLM (openai, groq, nvidia, or openrouter)
        model: Model to use for the agent
        temperature: Temperature for the LLM
        debug: Whether to enable debug logging

    Returns:
        Configured agent graph
    """
    # Define tools
    tools = [
        process_document_tool,
        read_markdown_file,
        edit_markdown_file,
        search_markdown_content,
        list_markdown_files,
        semantic_chunk_markdown,
        extract_triples_tool,
        refine_triples_tool,
    ]

    # Create system prompt
    system_prompt = """You are a helpful assistant specialized in document processing and markdown editing.

You have access to the following tools:
1. process_document_tool - Process documents (PDF, DOCX, PPTX, XLSX, images) and extract structured content
2. read_markdown_file - Read markdown files line by line
3. edit_markdown_file - Edit markdown files at specific lines
4. search_markdown_content - Search for content in markdown files
5. list_markdown_files - List all markdown files in a directory
6. semantic_chunk_markdown - Perform semantic chunking on markdown using LLM analysis
7. extract_triples_tool - Extract knowledge graph triples from chunks using LLM analysis
8. refine_triples_tool - Refine triples using Qdrant for entity resolution

When processing documents:
- Use the appropriate provider (nvidia or openrouter) and model
- Specify the content type (text, diagram, table, mixed) based on what you need to extract
- Use markdown format for best readability

When chunking markdown:
- Use semantic_chunk_markdown to split content based on topic shifts detected by LLM analysis
- The LLM reads the content and determines where topic changes occur based on semantic understanding
- Specify llm_provider and llm_model to control which LLM performs the chunking analysis
- Adjust similarity_threshold (0.0-1.0) to provide guidance to the LLM on chunk granularity
- Lower threshold = more chunks (finer granularity)
- Higher threshold = fewer chunks (coarser granularity)
- Output chunks contain only chunk_id and content for simplicity

When extracting triples:
- Use extract_triples_tool to extract knowledge graph triples from chunks
- The LLM analyzes each chunk and extracts triples with two-level predicate hierarchy
- Triples include subject, predicate, object, and temporal metadata
- Specify llm_provider and llm_model to control which LLM performs the triple extraction
- Output includes both per-chunk results and flattened all_triples array

When refining triples:
- Use refine_triples_tool to perform entity resolution using Qdrant
- The tool queries Qdrant for existing entities, predicates, and ontologies
- Uses LLM to determine canonical forms and identify synonyms
- Updates triples with canonical references for consistency
- Requires QDRANT_URL and QDRANT_API_KEY environment variables to be set
- Specify llm_provider and llm_model to control which LLM performs the canonical comparison

When editing markdown files:
- Always read the file first to understand its structure
- Use line numbers to make precise edits
- Use search to find specific content if needed
- Be careful with line numbers after insertions

Help users process their documents, make edits to markdown files, perform semantic chunking, extract knowledge graph triples, and refine triples for entity resolution efficiently."""

    # Create the LLM based on provider
    if provider == "openai":
        # Use OpenAI's API directly
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "groq":
        # Use Groq's API (OpenAI-compatible)
        groq_api_key = get_groq_api_key()
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    elif provider == "nvidia":
        # Use NVIDIA's API (OpenAI-compatible)
        nvidia_api_key = get_api_key()
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
    else:  # openrouter
        # Use OpenRouter's API
        openrouter_api_key = get_openrouter_api_key()
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    # Create the agent using the new API
    agent_graph = create_langchain_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        debug=debug,
    )

    return agent_graph


def run_agent_interactive(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    debug: bool = False,
) -> None:
    """Run the agent in interactive mode.

    Args:
        provider: API provider for document processing
        model: Model to use for the agent
        temperature: Temperature for the LLM
        debug: Whether to enable debug logging
    """
    print("🤖 LangChain Document Processing Agent")
    print("=" * 50)
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 50)

    # Create the agent
    agent_graph = create_agent(provider=provider, model=model, temperature=temperature, debug=debug)

    # Interactive loop
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! 👋")
                break

            if not user_input:
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Run the agent
            print("\nAgent: ", end="", flush=True)
            response = ""
            for chunk in agent_graph.stream({"messages": messages}, stream_mode="updates"):
                if "model" in chunk:
                    model_output = chunk["model"]
                    if "messages" in model_output:
                        for msg in model_output["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                print(msg.content, end="", flush=True)
                                response = msg.content  # Get the latest response
            print()

            # Add assistant response to messages
            if response:
                messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")


def run_agent_single_task(
    task: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    debug: bool = False,
) -> str:
    """Run the agent for a single task.

    Args:
        task: The task to perform
        provider: API provider for document processing
        model: Model to use for the agent
        temperature: Temperature for the LLM
        debug: Whether to enable debug logging

    Returns:
        Agent's response
    """
    agent_graph = create_agent(provider=provider, model=model, temperature=temperature, debug=debug)

    messages = [{"role": "user", "content": task}]
    response = ""

    for chunk in agent_graph.stream({"messages": messages}, stream_mode="updates"):
        if "model" in chunk:
            model_output = chunk["model"]
            if "messages" in model_output:
                for msg in model_output["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        response = msg.content  # Get the latest response

    return response if response else "No response generated"


if __name__ == "__main__":
    # Run interactive agent by default
    run_agent_interactive(debug=True)