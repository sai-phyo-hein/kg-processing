"""Workflow orchestrator for document processing and semantic chunking."""

import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.input_processor import DocumentProcessor
from kg_extractor.utils.markdown_formatter import save_markdown_result, save_text_markdown
from kg_extractor.utils.parser import (
    NVIDIAAPIError,
    NVIDIAConfig,
    OpenRouterAPIError,
    OpenRouterConfig,
    extract_text_from_document,
    extract_text_from_document_openrouter,
    get_api_key,
    get_openrouter_api_key,
    process_document_with_api,
    process_document_with_openrouter,
)
from kg_extractor.utils.semantic_chunker import chunk_markdown_file
from kg_extractor.utils.triple_extractor import extract_triples_from_chunks
from kg_extractor.utils.triple_refiner import refine_triples_from_file
from kg_extractor.utils.neo4j_graph_builder import build_graph_from_file


class DocumentChunkingWorkflow:
    """Orchestrates document processing, semantic chunking, and triple extraction workflow."""

    def __init__(
        self,
        provider: str = "nvidia",
        model: str = "microsoft/phi-4-multimodal-instruct",
        triplet_llm_provider: str = "openai",
        triplet_llm_model: str = "gpt-4o-mini",
    ):
        """Initialize the workflow orchestrator.

        Args:
            provider: API provider for document processing (nvidia or openrouter)
            model: Model to use for document processing
            triplet_llm_provider: LLM provider for triple extraction (openai, groq, nvidia, openrouter)
            triplet_llm_model: Model to use for triple extraction
        """
        self.provider = provider
        self.model = model
        self.triplet_llm_provider = triplet_llm_provider
        self.triplet_llm_model = triplet_llm_model
        self.output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    def run_workflow(
        self,
        input_file: str,
        content_type: str = "mixed",
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        output_format: str = "markdown",
        chunking_llm_provider: str = "openai",
        chunking_llm_model: str = "gpt-4o-mini",
        refine_triples: bool = True,
        refinement_llm_provider: str = "openai",
        refinement_llm_model: str = "gpt-4o-mini",
        build_graph: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete workflow: process document, chunk semantically, extract triples, refine, and build graph.

        Args:
            input_file: Path to input document
            content_type: Type of content to extract (text, diagram, table, mixed)
            similarity_threshold: Threshold for topic change detection (0.0-1.0)
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            output_format: Output format for document processing (text, json, markdown)
            chunking_llm_provider: LLM provider for chunking (openai, groq, nvidia, openrouter)
            chunking_llm_model: Model to use for chunking analysis
            refine_triples: Whether to refine triples using Qdrant
            refinement_llm_provider: LLM provider for triple refinement
            refinement_llm_model: Model for triple refinement
            build_graph: Whether to build Neo4j graph from refined triples

        Returns:
            Dictionary with processing results, chunking results, triple results, refinement results, and graph results
        """
        try:
            # Step 1: Process document
            markdown_path = self._process_document(input_file, content_type, output_format)

            # Step 2: Semantic chunking
            chunks_path = self._chunk_markdown(
                markdown_path,
                similarity_threshold,
                min_chunk_size,
                max_chunk_size,
                chunking_llm_provider,
                chunking_llm_model,
            )

            # Step 3: Triplet extraction
            triplets_path = self._extract_triplets(chunks_path, input_file)

            # Step 4: Triple refinement (optional)
            refined_path = None
            if refine_triples:
                refined_path = self._refine_triples(
                    triplets_path,
                    refinement_llm_provider,
                    refinement_llm_model,
                )

            # Step 5: Neo4j graph building (optional)
            graph_stats = None
            if build_graph and refined_path:
                graph_stats = self._build_graph(refined_path)

            return {
                "markdown_output": markdown_path,
                "chunks_output": chunks_path,
                "triples_output": triplets_path,
                "refined_output": refined_path,
                "graph_stats": graph_stats,
                "status": "success",
                "error": None,
            }

        except (NVIDIAAPIError, OpenRouterAPIError) as e:
            return {
                "markdown_output": None,
                "chunks_output": None,
                "triples_output": None,
                "refined_output": None,
                "graph_stats": None,
                "status": "error",
                "error": f"API Error: {e}",
            }
        except Exception as e:
            return {
                "markdown_output": None,
                "chunks_output": None,
                "triples_output": None,
                "refined_output": None,
                "graph_stats": None,
                "status": "error",
                "error": f"Error: {e}",
            }

    def _process_document(
        self, file_path: str, content_type: str = "mixed", output_format: str = "markdown"
    ) -> str:
        """Process a document file and extract structured content.

        Args:
            file_path: Path to the document file
            content_type: Type of content to focus on
            output_format: Output format (text, json, markdown)

        Returns:
            Path to the processed output file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
            NVIDIAAPIError: If NVIDIA API fails
            OpenRouterAPIError: If OpenRouter API fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file type
        file_type = DocumentProcessor.get_file_type(file_path)
        if file_type == "unknown":
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Get API key and create configuration
        if self.provider == "nvidia":
            api_key = get_api_key()
            config = NVIDIAConfig(
                api_key=api_key,
                model=self.model,
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )
        else:  # openrouter
            api_key = get_openrouter_api_key()
            config = OpenRouterConfig(
                api_key=api_key,
                model=self.model,
                max_tokens=4096,
                temperature=0.2,
                top_p=0.7,
                stream=False,
            )

        # Process document based on output format
        if output_format == "json":
            if self.provider == "nvidia":
                result = extract_text_from_document(file_path, config, content_type)
            else:
                result = extract_text_from_document_openrouter(file_path, config, content_type)

            # Save JSON result
            output_file = Path("output") / f"{path.stem}_analysis.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            import json

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            return str(output_file)

        elif output_format == "markdown":
            if self.provider == "nvidia":
                result = process_document_with_api(file_path, config, content_type)
            else:
                result = process_document_with_openrouter(file_path, config, content_type)

            # Save markdown result
            output_file = save_markdown_result(result, file_path)
            return str(output_file)

        else:  # text format
            if self.provider == "nvidia":
                text = extract_text_from_document(file_path, config, content_type)
            else:
                text = extract_text_from_document_openrouter(file_path, config, content_type)

            # Save text result
            output_file = save_text_markdown(text, file_path)
            return str(output_file)

    def _chunk_markdown(
        self,
        markdown_path: str,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunking_llm_provider: str = "openai",
        chunking_llm_model: str = "gpt-4o-mini",
    ) -> str:
        """Perform semantic chunking on a markdown file using LLM analysis.

        Args:
            markdown_path: Path to the markdown file to chunk
            similarity_threshold: Threshold for detecting topic changes
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            chunking_llm_provider: LLM provider for chunking
            chunking_llm_model: Model to use for chunking analysis

        Returns:
            Path to the chunks output file

        Raises:
            FileNotFoundError: If markdown file doesn't exist
            Exception: If chunking fails
        """
        path = Path(markdown_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

        # Perform chunking
        output_path = chunk_markdown_file(
            file_path=markdown_path,
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            llm_provider=chunking_llm_provider,
            llm_model=chunking_llm_model,
            output_dir=self.output_dir,
        )

        return output_path

    def _extract_triplets(
        self,
        chunks_path: str,
        input_file: str,
    ) -> str:
        """Extract triples from chunks using LLM analysis.

        Args:
            chunks_path: Path to the chunks JSON file
            input_file: Original input file path for context

        Returns:
            Path to the triples output file

        Raises:
            FileNotFoundError: If chunks file doesn't exist
            Exception: If triple extraction fails
        """
        path = Path(chunks_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        # Read chunks
        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = chunks_data.get("chunks", [])

        # Extract triples
        output_path = extract_triples_from_chunks(
            chunks=chunks,
            source_file=input_file,
            llm_provider=self.triplet_llm_provider,
            llm_model=self.triplet_llm_model,
            output_dir=self.output_dir,
        )

        return output_path

    def _refine_triples(
        self,
        triples_path: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ) -> str:
        """Refine triples using Qdrant for entity resolution.

        Args:
            triples_path: Path to the triples JSON file
            llm_provider: LLM provider for canonical comparison
            llm_model: Model for canonical comparison

        Returns:
            Path to the refined triples file

        Raises:
            FileNotFoundError: If triples file doesn't exist
            Exception: If refinement fails
        """
        path = Path(triples_path)
        if not path.exists():
            raise FileNotFoundError(f"Triples file not found: {triples_path}")

        # Refine triples
        output_path = refine_triples_from_file(
            input_path=triples_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

        return output_path

    def _build_graph(
        self,
        refined_path: str,
    ) -> Dict[str, Any]:
        """Build Neo4j graph from refined triples.

        Args:
            refined_path: Path to the refined triples JSON file

        Returns:
            Statistics about the graph building process

        Raises:
            FileNotFoundError: If refined triples file doesn't exist
            Exception: If graph building fails
        """
        path = Path(refined_path)
        if not path.exists():
            raise FileNotFoundError(f"Refined triples file not found: {refined_path}")

        # Build graph
        stats = build_graph_from_file(refined_path)

        return stats


def run_workflow(
    input_file: str,
    provider: str = "nvidia",
    model: str = "microsoft/phi-4-multimodal-instruct",
    content_type: str = "mixed",
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
    output_format: str = "markdown",
    chunking_llm_provider: str = "openai",
    chunking_llm_model: str = "gpt-4o-mini",
    triplet_llm_provider: str = "openai",
    triplet_llm_model: str = "gpt-4o-mini",
    refine_triples: bool = True,
    refinement_llm_provider: str = "openai",
    refinement_llm_model: str = "gpt-4o-mini",
    build_graph: bool = True,
) -> Dict[str, Any]:
    """Run the complete workflow: process document, chunk semantically, extract triples, refine, and build graph.

    This is a convenience function that creates a workflow orchestrator
    and runs it with the specified parameters.

    Args:
        input_file: Path to input document
        provider: API provider for document processing
        model: Model to use for document processing
        content_type: Type of content to extract
        similarity_threshold: Threshold for topic change detection
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
        output_format: Output format for document processing
        chunking_llm_provider: LLM provider for chunking
        chunking_llm_model: Model to use for chunking analysis
        triplet_llm_provider: LLM provider for triple extraction
        triplet_llm_model: Model to use for triple extraction
        refine_triples: Whether to refine triples using Qdrant
        refinement_llm_provider: LLM provider for triple refinement
        refinement_llm_model: Model for triple refinement
        build_graph: Whether to build Neo4j graph from refined triples

    Returns:
        Dictionary with processing results, chunking results, triple results, refinement results, and graph results
    """
    workflow = DocumentChunkingWorkflow(
        provider=provider,
        model=model,
        triplet_llm_provider=triplet_llm_provider,
        triplet_llm_model=triplet_llm_model,
    )
    return workflow.run_workflow(
        input_file=input_file,
        content_type=content_type,
        similarity_threshold=similarity_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        output_format=output_format,
        chunking_llm_provider=chunking_llm_provider,
        chunking_llm_model=chunking_llm_model,
        refine_triples=refine_triples,
        refinement_llm_provider=refinement_llm_provider,
        refinement_llm_model=refinement_llm_model,
        build_graph=build_graph,
    )
