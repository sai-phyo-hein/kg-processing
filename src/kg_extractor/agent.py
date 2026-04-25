"""LangChain agent for document processing and markdown editing."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent as create_langchain_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from kg_extractor.input_processor import DocumentProcessor
from kg_extractor.markdown_formatter import save_markdown_result, save_text_markdown
from kg_extractor.parser import (
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
    ]

    # Create system prompt
    system_prompt = """You are a helpful assistant specialized in document processing and markdown editing.

You have access to the following tools:
1. process_document_tool - Process documents (PDF, DOCX, PPTX, XLSX, images) and extract structured content
2. read_markdown_file - Read markdown files line by line
3. edit_markdown_file - Edit markdown files at specific lines
4. search_markdown_content - Search for content in markdown files
5. list_markdown_files - List all markdown files in a directory

When processing documents:
- Use the appropriate provider (nvidia or openrouter) and model
- Specify the content type (text, diagram, table, mixed) based on what you need to extract
- Use markdown format for best readability

When editing markdown files:
- Always read the file first to understand its structure
- Use line numbers to make precise edits
- Use search to find specific content if needed
- Be careful with line numbers after insertions

Help users process their documents and make edits to markdown files efficiently."""

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