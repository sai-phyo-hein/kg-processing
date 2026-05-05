"""Document processing node for LangGraph workflow."""

from pathlib import Path
from typing import TYPE_CHECKING

from kg_extractor.workflow.functions.process_document import (
    DocumentProcessor,
    save_markdown_result,
    NVIDIAConfig,
    OpenRouterConfig,
    OpenAIConfig,
    GoogleConfig,
    get_api_key,
    get_openrouter_api_key,
    get_openai_api_key,
    get_google_api_key,
    process_document_with_nvidia,
    process_document_with_openrouter,
    process_document_with_openai,
    process_document_with_google,
)

if TYPE_CHECKING:
    from ..langgraph_workflow import WorkflowState


def process_document_node(state: "WorkflowState") -> "WorkflowState":
    """Process document and extract structured content."""
    try:
        print("📄 Processing document...")

        path = Path(state["input_file"])
        if not path.exists():
            raise FileNotFoundError(f"File not found: {state['input_file']}")

        # Get file type
        file_type = DocumentProcessor.get_file_type(state["input_file"])
        if file_type == "unknown":
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Get API key and create configuration
        if state["provider"] == "nvidia":
            api_key = get_api_key()
            config = NVIDIAConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        elif state["provider"] == "openrouter":
            api_key = get_openrouter_api_key()
            config = OpenRouterConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        elif state["provider"] == "openai":
            api_key = get_openai_api_key()
            config = OpenAIConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        else:  # google
            api_key = get_google_api_key()
            config = GoogleConfig(
                api_key=api_key,
                model=state["model"],
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )

        # Process document and save as markdown
        if state["provider"] == "nvidia":
            result = process_document_with_nvidia(
                state["input_file"], config, pages=state["pages"]
            )
        elif state["provider"] == "openrouter":
            result = process_document_with_openrouter(
                state["input_file"], config, pages=state["pages"]
            )
        elif state["provider"] == "openai":
            result = process_document_with_openai(
                state["input_file"], config, pages=state["pages"]
            )
        else:  # google
            result = process_document_with_google(
                state["input_file"], config, pages=state["pages"]
            )

        output_file = save_markdown_result(result, state["input_file"])
        state["markdown_path"] = output_file

        state["current_step"] = "extract_metadata"
        print(f"✅ Document processed: {state['markdown_path']}")

    except Exception as e:
        state["status"] = "error"
        state["error"] = f"Document processing failed: {e}"
        print(f"❌ Error: {e}")

    return state
