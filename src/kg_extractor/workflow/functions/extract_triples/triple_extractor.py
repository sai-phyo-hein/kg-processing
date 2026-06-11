"""Triplet extraction module for knowledge graph construction."""

import json
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.model_setup import TRIPLET_PROVIDER, TRIPLET_MODEL, get_reasoning_llm
from kg_extractor.utils.prompts import (
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    create_triple_extraction_user_message,
)
from kg_extractor.utils.llm_response_parser import parse_triple_extraction_response
from kg_extractor.workflow.functions.chunk_document.semantic_chunker import SemanticChunker

# Try to import faster JSON library
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False


class TripleExtractor:
    """Extract knowledge graph triples from text chunks using LLM analysis.
    
    Consumes chunks produced by SemanticChunker:
    {
        "chunk_id": int,
        "content": str
    }
    """

    def __init__(
        self,
        llm_provider: str = TRIPLET_PROVIDER,
        llm_model: str = TRIPLET_MODEL,
        batch_size: int = 1,
        max_workers: int = 20,
    ):
        """Initialize the triple extractor.

        Args:
            llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
            llm_model: Model to use for LLM analysis
            batch_size: Number of chunks to batch together in one LLM call (>1 for batching)
            max_workers: Maximum number of concurrent worker threads
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)
        self._response_cache = {}  # Cache for identical chunks
        
        # Initialize stateful agent with persistent system prompt
        self._init_stateful_agent()

    def _init_stateful_agent(self):
        """Initialize stateful agent with persistent system prompt.
        
        The triple extraction agent maintains the system prompt between calls.
        User messages (chunk content) are cleared after each response.
        """
        from langchain_core.messages import SystemMessage
        from langchain_core.chat_history import InMemoryChatMessageHistory
        
        # Create LLM instance and history
        # Use high max_tokens for triple extraction: 40-50 triples with long evidence quotes
        # can easily require many tokens in Thai (which uses more tokens per character)
        # Claude Sonnet 4.6 supports up to 128K output tokens
        self.extraction_agent = {
            'llm': get_reasoning_llm(model=self.llm_model, temperature=0.3, max_tokens=64000),
            'history': InMemoryChatMessageHistory()
        }
        
        # Add persistent system message
        self.extraction_agent['history'].add_message(
            SystemMessage(content=TRIPLE_EXTRACTION_SYSTEM_PROMPT)
        )

    def extract_triples_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        source_file: str = "",
    ) -> List[Dict[str, Any]]:
        """Extract triples from multiple chunks in parallel.
        
        Expects chunks in SemanticChunker format:
        {
            "chunk_id": int,
            "content": str
        }

        Args:
            chunks: List of chunk dictionaries from SemanticChunker.chunk_markdown()
            source_file: Source file path for context

        Returns:
            List of triple extraction results
        """
        if not chunks:
            return []

        # Build result map to maintain chunk ordering
        chunk_id_to_index = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
        results = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._extract_triples_from_single_chunk,
                    chunk,
                    source_file,
                ): chunk["chunk_id"]
                for chunk in chunks
            }

            # Collect results as they complete (no need to sort later)
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    # Place result at original chunk index
                    results[chunk_id_to_index[chunk_id]] = result
                except Exception as e:
                    print(f"Warning: Failed to extract triples from chunk {chunk_id}: {e}")
                    # Add empty result for failed chunks at correct position
                    results[chunk_id_to_index[chunk_id]] = {
                        "chunk_id": chunk_id,
                        "triples": [],
                        "error": str(e),
                    }

        # Filter out None values (shouldn't happen, but be safe)
        results = [r for r in results if r is not None]
        return results

    def _extract_triples_from_single_chunk(
        self,
        chunk: Dict[str, Any],
        source_file: str = "",
    ) -> Dict[str, Any]:
        """Extract triples from a single chunk.

        Args:
            chunk: Chunk dictionary with chunk_id and content
            source_file: Source file path for context

        Returns:
            Dictionary with chunk_id, document_metadata, and extracted triples
        """
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        # Number the content lines for LLM input
        numbered_content, original_lines = self._number_chunk_content(content)

        # Create user message with numbered content
        user_message = create_triple_extraction_user_message(
            numbered_content, source_file, chunk_id
        )

        # Get LLM response from stateful agent
        response = self._get_llm_response(user_message)

        # Parse response to extract triples and metadata
        parsed_data = self._parse_llm_response(response)

        # Post-process: resolve evidence_lines → evidence_quote from source
        triples = parsed_data.get("discovered_triples", [])
        triples = self._resolve_evidence_lines(triples, original_lines)

        return {
            "chunk_id": chunk_id,
            "document_metadata": parsed_data.get("document_metadata", {}),
            "triples": triples,
        }

    def _get_llm_response(self, user_message: str) -> str:
        """Get response from stateful agent with persistent system prompt.
        
        The system prompt persists in the agent's history.
        User messages are added, processed, then cleared after each call.

        Args:
            user_message: User-specific content (source + chunk text)

        Returns:
            LLM response text
        """
        try:
            # Get the agent
            agent = self.extraction_agent
            
            # Add user message to agent's history (system message already persists)
            agent['history'].add_user_message(user_message)
            
            # Get all messages (system + user)
            messages = agent['history'].messages
            
            # Invoke the agent's LLM
            response = agent['llm'].invoke(messages)
            
            # Clear user message to prevent accumulation (keep only system message)
            while len(agent['history'].messages) > 1:  # Keep only the first message (system)
                agent['history'].messages.pop()
            
            return response.content

        except Exception as e:
            # Fallback: return empty response
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"document_metadata": {"reference_date": null, "source_id": null, "chunk_id": null}, "discovered_triples": []}'

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract triples and metadata."""
        return parse_triple_extraction_response(response)

    @staticmethod
    def _number_chunk_content(content: str) -> tuple:
        """Split chunk content into lines and build numbered text for LLM.

        Returns:
            Tuple of (numbered_text, original_lines) where:
            - numbered_text: string with [NNNN] prefix per line
            - original_lines: list of original line strings (for post-processing)
        """
        lines = content.split("\n")
        numbered_text = SemanticChunker._build_numbered_content(lines, start_number=1)
        return numbered_text, lines

    @staticmethod
    def _resolve_evidence_lines(
        triples: list, original_lines: list
    ) -> list:
        """Replace evidence_lines with evidence_quote from source text.

        For each triple that has evidence_lines in properties:
        1. Extract lines from original_lines using start/end (1-based, inclusive).
        2. Join them into evidence_quote.
        3. Remove evidence_lines from properties.
        4. Keep evidence_quote_en as-is (provided by LLM for Thai sources).

        Args:
            triples: List of triple dicts from LLM response.
            original_lines: Original un-numbered lines from chunk content.

        Returns:
            The same triples list with evidence_quote populated.
        """
        n_lines = len(original_lines)
        for triple in triples:
            props = triple.get("properties", {})
            evidence_lines = props.pop("evidence_lines", None)

            if evidence_lines and isinstance(evidence_lines, dict):
                start = int(evidence_lines.get("start", 1))
                end = int(evidence_lines.get("end", start))

                # Clamp to valid range (1-based, inclusive)
                start = max(1, min(start, n_lines))
                end = max(start, min(end, n_lines))

                # Extract text (convert 1-based to 0-based index)
                extracted = original_lines[start - 1:end]
                props["evidence_quote"] = "\n".join(
                    line.rstrip("\n") for line in extracted
                ).strip()
            elif "evidence_quote" not in props:
                # Fallback: LLM returned neither evidence_lines nor evidence_quote
                props["evidence_quote"] = ""

        return triples

    def save_triples(
        self,
        triple_results: List[Dict[str, Any]],
        output_path: str,
        community_id: str = None,
    ) -> str:
        """Save triples to a JSON file.

        Args:
            triple_results: List of triple extraction results
            output_path: Path to save the triples
            community_id: Optional community ID to add to each triple's properties

        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Update triples in chunks and flatten all triples
        all_triples = []
        for result in triple_results:
            chunk_id = result["chunk_id"]
            for triple in result.get("triples", []):
                # Add chunk_id to each triple for backward compatibility
                triple["chunk_id"] = chunk_id

                # Add community_id to properties if provided
                if community_id:
                    if "properties" not in triple:
                        triple["properties"] = {}
                    triple["properties"]["community_id"] = community_id

                all_triples.append(triple)

        # Create output structure
        output_data = {
            "source_file": str(output_file),
            "total_chunks": len(triple_results),
            "total_triples": len(all_triples),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "chunks": triple_results,
            "all_triples": all_triples,
        }

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return str(output_file)


def extract_triples_from_chunks(
    chunks: List[Dict[str, Any]],
    source_file: str = "",
    llm_provider: str = TRIPLET_PROVIDER,
    llm_model: str = TRIPLET_MODEL,
    output_dir: str = None,
    community_id: str = None,
    batch_size: int = 1,
    max_workers: int = 20,
) -> str:
    """Extract triples from chunks using LLM analysis and save results.

    Args:
        chunks: List of chunk dictionaries with chunk_id and content
        source_file: Source file path for context
        llm_provider: LLM provider to use
        llm_model: Model to use for LLM analysis
        output_dir: Directory to save triples (default: project root/output)
        community_id: Optional community ID to add to each triple's properties
        batch_size: Number of chunks to batch in one LLM call (experimental, set >1 to batch)
        max_workers: Maximum concurrent workers (default 20, increase for more parallelism)

    Returns:
        Path to saved triples file
    """
    # Set default output directory to project root/output
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent.parent.parent / "output")
    # Create extractor
    extractor = TripleExtractor(
        llm_provider=llm_provider,
        llm_model=llm_model,
        batch_size=batch_size,
        max_workers=max_workers,
    )

    # Extract triples from chunks
    triple_results = extractor.extract_triples_from_chunks(chunks, source_file)

    # Generate output path
    if source_file:
        input_path = Path(source_file)
        output_filename = f"{input_path.stem}_triples.json"
    else:
        output_filename = "extracted_triples.json"
    output_path = Path(output_dir) / output_filename

    # Save triples with community_id
    result_path = extractor.save_triples(triple_results, str(output_path), community_id)

    return result_path
