"""Triplet extraction module for knowledge graph construction.

This module handles only triple *extraction*: a stateless LLM call per chunk
(E1) and evidence-line resolution into ``evidence_quote`` (E2).

Thai->English translation of the extracted fields now lives in its own pipeline
node — see :mod:`kg_extractor.workflow.functions.translate_triples.translator`,
which runs after this node and before refinement.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.model_setup import (
    TRIPLET_PROVIDER,
    TRIPLET_MODEL,
    get_reasoning_llm,
)
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

    Thread safety
    -------------
    Each call to _get_llm_response builds a fresh [SystemMessage, HumanMessage]
    list and invokes the LLM with no shared mutable state, so it is safe to call
    from multiple threads simultaneously (E1 fix).
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

        # E1 fix: single shared, stateless LLM instance (no chat history).
        # max_tokens=24000: sized to match the 80-triple hard cap in the prompt.
        self._llm = get_reasoning_llm(model=self.llm_model, temperature=0.3, max_tokens=24000)

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
        total = len(chunks)
        completed = 0

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
                    results[chunk_id_to_index[chunk_id]] = result
                except Exception as e:
                    print(f"Warning: Failed to extract triples from chunk {chunk_id}: {e}")
                    results[chunk_id_to_index[chunk_id]] = {
                        "chunk_id": chunk_id,
                        "triples": [],
                        "error": str(e),
                    }
                finally:
                    # Progress logging (single-threaded here) so a slow run is
                    # visibly making progress rather than appearing to hang.
                    completed += 1
                    print(f"  [{completed}/{total}] chunk {chunk_id} done")

        # Filter out None values (shouldn't happen, but be safe)
        results = [r for r in results if r is not None]
        return results

    def _extract_triples_from_single_chunk(
        self,
        chunk: Dict[str, Any],
        source_file: str = "",
    ) -> Dict[str, Any]:
        """Extract triples from a single chunk.

        Runs entirely within one outer-pool worker thread: extraction call,
        evidence resolution, and (for Thai chunks) batched translation.

        Args:
            chunk: Chunk dictionary with chunk_id and content
            source_file: Source file path for context

        Returns:
            Dictionary with chunk_id and extracted triples
        """
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        # Number the content lines for LLM input
        numbered_content, original_lines = self._number_chunk_content(content)

        # Create user message with numbered content
        user_message = create_triple_extraction_user_message(
            numbered_content, source_file, chunk_id
        )

        # Get LLM response (stateless call)
        response = self._get_llm_response(user_message)

        # Parse response to extract triples and metadata
        parsed_data = self._parse_llm_response(response)

        # Post-process: resolve evidence_lines -> evidence_quote from source
        triples = parsed_data.get("discovered_triples", [])
        triples = self._resolve_evidence_lines(triples, original_lines)

        # NOTE: Thai->English translation of the _en fields is intentionally NOT
        # done here.  It runs in the dedicated translate_triples pipeline node
        # (see workflow/functions/translate_triples/translator.py), which
        # translates every chunk that has triples.
        return {
            "chunk_id": chunk_id,
            "triples": triples,
        }

    def _get_llm_response(self, user_message: str) -> str:
        """Thread-safe LLM call (E1 fix).

        Builds a fresh [SystemMessage, HumanMessage] list on every call so
        multiple threads can invoke this simultaneously without any shared
        mutable state.

        Args:
            user_message: User-specific content (source + chunk text)

        Returns:
            LLM response text
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            messages = [
                SystemMessage(content=TRIPLE_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"discovered_triples": []}'

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract triples and metadata."""
        return parse_triple_extraction_response(response)

    # ------------------------------------------------------------------ #
    # Content numbering / evidence resolution
    # ------------------------------------------------------------------ #

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
    def _resolve_evidence_lines(triples: list, original_lines: list) -> list:
        """Replace evidence_lines with evidence_quote from source text (E2 fix).

        For each triple that has evidence_lines in properties:
        1. Enforce the minimum 3-line span required by the extraction prompt.
        2. Extract lines from original_lines using start/end (1-based, inclusive).
        3. Join them into evidence_quote.
        4. Remove evidence_lines from properties.

        Args:
            triples: List of triple dicts from LLM response.
            original_lines: Original un-numbered lines from chunk content.

        Returns:
            The same triples list with evidence_quote populated.
        """
        MIN_SPAN = 3  # mirrors the prompt's MINIMUM requirement

        n_lines = len(original_lines)
        for triple in triples:
            props = triple.get("properties", {})
            evidence_lines = props.pop("evidence_lines", None)

            if evidence_lines and isinstance(evidence_lines, dict):
                start = int(evidence_lines.get("start", 1))
                end = int(evidence_lines.get("end", start))

                # Enforce minimum span before clamping
                if (end - start + 1) < MIN_SPAN:
                    end = start + MIN_SPAN - 1

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

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #

    @staticmethod
    def _stamp_triple_results(
        triple_results: List[Dict[str, Any]],
        community_id: str = None,
    ) -> int:
        """Stamp each triple with its chunk_id (and optional community_id).

        Mutates triples in place. Returns the total number of triples stamped.
        """
        total_triples = 0
        for result in triple_results:
            chunk_id = result["chunk_id"]
            for triple in result.get("triples", []):
                triple["chunk_id"] = chunk_id
                if community_id:
                    if "properties" not in triple:
                        triple["properties"] = {}
                    triple["properties"]["community_id"] = community_id
                total_triples += 1
        return total_triples

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

        # Stamp each triple with chunk_id and optional community_id.
        # The redundant top-level all_triples flat list is intentionally omitted;
        # downstream consumers read from chunks[].triples (E3).
        total_triples = self._stamp_triple_results(triple_results, community_id)

        output_data = {
            "source_file": str(output_file),
            "total_chunks": len(triple_results),
            "total_triples": total_triples,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "chunks": triple_results,
        }

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
    chunk_id: int = None,
) -> str:
    """Extract triples from chunks using LLM analysis and save results.

    Thai->English translation is NOT performed here — it runs in the dedicated
    translate_triples node downstream.

    Args:
        chunks: List of chunk dictionaries with chunk_id and content
        source_file: Source file path for context
        llm_provider: LLM provider to use
        llm_model: Model to use for LLM analysis
        output_dir: Directory to save triples (default: project root/output)
        community_id: Optional community ID to add to each triple's properties
        batch_size: Number of chunks to batch in one LLM call (experimental)
        max_workers: Maximum concurrent workers
        chunk_id: Optional single chunk_id to (re-)extract. When set, only that
            chunk is processed and its result is merged back into the existing
            triples file in place (all other chunks preserved). If no prior file
            exists, a fresh file containing only that chunk is written.

    Returns:
        Path to saved triples file
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent.parent.parent / "output")

    extractor = TripleExtractor(
        llm_provider=llm_provider,
        llm_model=llm_model,
        batch_size=batch_size,
        max_workers=max_workers,
    )

    if source_file:
        output_filename = f"{Path(source_file).stem}_triples.json"
    else:
        output_filename = "extracted_triples.json"
    output_path = Path(output_dir) / output_filename

    # --- Single-chunk mode: extract one chunk, merge into the existing file ---
    if chunk_id is not None:
        targets = [c for c in chunks if c.get("chunk_id") == chunk_id]
        if not targets:
            raise ValueError(
                f"chunk_id {chunk_id} not found among {len(chunks)} chunks"
            )
        triple_results = extractor.extract_triples_from_chunks(targets, source_file)

        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            # Stamp the re-extracted chunk's triples (chunk_id + community_id),
            # matching what save_triples does on a full run.
            extractor._stamp_triple_results(triple_results, community_id)

            existing_chunks = existing.get("chunks", [])
            replaced = False
            for i, c in enumerate(existing_chunks):
                if c.get("chunk_id") == chunk_id:
                    existing_chunks[i] = triple_results[0]
                    replaced = True
                    break
            if not replaced:
                existing_chunks.append(triple_results[0])
                existing_chunks.sort(key=lambda c: c.get("chunk_id", 0))

            existing["chunks"] = existing_chunks
            existing["total_chunks"] = len(existing_chunks)
            existing["total_triples"] = sum(
                len(c.get("triples", [])) for c in existing_chunks
            )
            existing["llm_provider"] = llm_provider
            existing["llm_model"] = llm_model
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            return str(output_path)

        # No prior file: write a fresh file containing only this chunk.
        return extractor.save_triples(triple_results, str(output_path), community_id)

    # --- Default mode: extract all chunks, overwrite the output file ---
    triple_results = extractor.extract_triples_from_chunks(chunks, source_file)
    result_path = extractor.save_triples(triple_results, str(output_path), community_id)
    return result_path