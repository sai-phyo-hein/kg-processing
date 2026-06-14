"""Triplet extraction module for knowledge graph construction."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.model_setup import TRIPLET_PROVIDER, TRIPLET_MODEL, TRANSLATION_MODEL, get_reasoning_llm
from kg_extractor.utils.prompts import (
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
    create_triple_extraction_user_message,
    create_translation_user_message,
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
    list and invokes the LLM with no shared mutable state.  It is therefore safe
    to call from multiple threads simultaneously (E1 fix).
    """

    def __init__(
        self,
        llm_provider: str = TRIPLET_PROVIDER,
        llm_model: str = TRIPLET_MODEL,
        translation_model: str = TRANSLATION_MODEL,
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
        self.translation_model = translation_model
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)

        # E1 fix: single shared LLM instance (stateless — no chat history).
        # max_tokens=24000: sized to match the 80-triple hard cap in the prompt.
        # At ~205 tokens/triple (Thai JSON, no _en fields) + 80 wrapper:
        #   80 triples × 205 + 80 = 16,480 tokens → 24,000 gives 45% headroom.
        # The prompt enforces the 80-triple cap; max_tokens is a safety net
        # against prompt non-compliance, not the effective limit.
        self._llm = get_reasoning_llm(model=self.llm_model, temperature=0.3, max_tokens=24000)

        # Separate cheap model for Thai→English translation.
        # All _en fields generated here, NOT by the expensive extraction model.
        # Per-triple output budget:
        #   names × 3 (~15 tokens) + predicate (~5) + attributes (~20)
        #   + evidence_quote_en (~200 tokens for a typical Thai paragraph)
        #   + JSON structure overhead (~30 tokens)
        #   = ~270 tokens realistic worst case → 1024 gives safe 4× headroom.
        # (512 caused truncation on long evidence quotes.)
        self._translation_llm = get_reasoning_llm(
            model=self.translation_model,
            temperature=0.1,
            max_tokens=1024,
        )

        # Rate-limit guard for translation calls.
        # claude-3-haiku's RPM limit means we cannot fire hundreds of calls at once.
        # This semaphore caps the number of in-flight translation requests across ALL
        # concurrent chunk threads.  Adjust translation_max_concurrent to stay under
        # your tier's RPM limit (e.g. 50 concurrent ≈ ~200 RPM at ~0.25s per call).
        translation_max_concurrent = 50
        self._translation_semaphore = threading.Semaphore(translation_max_concurrent)

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

        # Post-process: add all _en fields via cheap translation model.
        # Only runs when the detected language is Thai (th / mixed-th).
        detected_lang = parsed_data.get("document_metadata", {}).get("detected_language", "en")
        # Translate for any detected language that contains Thai script.
        # "mixed-en" chunks (Thai tables with "not specified" English values)
        # still have Thai entity names that need _en fields.
        if detected_lang in ("th", "mixed-th", "mixed-en"):
            triples = self._translate_chunk_triples(triples)

        return {
            "chunk_id": chunk_id,
            "document_metadata": parsed_data.get("document_metadata", {}),
            "triples": triples,
        }

    def _get_llm_response(self, user_message: str) -> str:
        """Thread-safe LLM call.

        Builds a fresh [SystemMessage, HumanMessage] list on every call so
        multiple threads can invoke this simultaneously without any shared
        mutable state (E1 fix — no InMemoryChatMessageHistory).

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
            return '{"document_metadata": {"reference_date": null, "source_id": null, "chunk_id": null}, "discovered_triples": []}'

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract triples and metadata."""
        return parse_triple_extraction_response(response)

    def _get_translation_response(self, user_message: str) -> str:
        """Thread-safe translation call using the cheap Haiku model.

        Acquires the shared semaphore before calling the API so the total number
        of in-flight translation requests stays within the model's RPM limit.
        Retries up to 5 times with exponential backoff on rate-limit (429) errors.

        Args:
            user_message: JSON array of translation items (from
                create_translation_user_message).

        Returns:
            Raw LLM response text (expected to be a JSON array).
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=TRANSLATION_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        max_attempts = 5
        wait = 5.0  # seconds before first retry

        for attempt in range(1, max_attempts + 1):
            try:
                with self._translation_semaphore:
                    response = self._translation_llm.invoke(messages)
                return response.content

            except Exception as e:
                err = str(e).lower()
                is_rate_limit = "429" in err or "rate limit" in err or "rate_limit" in err or "too many" in err

                if is_rate_limit and attempt < max_attempts:
                    print(f"Warning: Translation rate-limited (attempt {attempt}/{max_attempts}), "
                          f"retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    wait = min(wait * 2, 60)  # exponential backoff, cap at 60s
                else:
                    if not is_rate_limit:
                        print(f"Warning: Failed to get translation response: {e}")
                    else:
                        print(f"Warning: Translation rate-limited after {max_attempts} attempts, skipping.")
                    return "[]"

        return "[]"

    def _translate_single_triple(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Translate one triple's Thai fields via a single cheap Haiku call.

        Each call is tiny (~30-50 output tokens for names/predicate/attributes
        plus ~100-200 for evidence_quote_en), well within claude-3-haiku's
        4,096 token output limit.

        Args:
            item: Dict with id + Thai fields (subject_th, predicate_th, etc.)

        Returns:
            Dict with id + translated _en fields, or just {"id": item["id"]}
            on failure so the caller can skip gracefully.
        """
        import re as _re

        user_message = create_translation_user_message([item])
        raw_response = self._get_translation_response(user_message)

        try:
            text = raw_response.strip()
            # Strip markdown fences if present
            text = _re.sub(r"^```[a-z]*\n?", "", text)
            text = _re.sub(r"\n?```$", "", text.strip())

            # Repair: Haiku sometimes outputs ALL_CAPS_SNAKE_CASE predicate values
            # without quotes — e.g.  "predicate_en": HAS_DETAILS
            # Fix by quoting any unquoted ALL_CAPS_SNAKE_CASE value after a colon.
            text = _re.sub(
                r':\s*([A-Z][A-Z0-9_]{2,})(\s*[,}\]])',
                r': "\1"\2',
                text,
            )

            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                return parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"Warning: Failed to parse translation for triple id={item.get('id')}: {e}")
            print(f"  Raw response (first 300 chars): {repr(raw_response[:300])}")
        return {"id": item.get("id")}

    def _translate_chunk_triples(self, triples: list) -> list:
        """Translate all Thai fields in a chunk's triples.

        Fires one Haiku call per triple, all concurrently via ThreadPoolExecutor
        (reusing the same max_workers as extraction).  Per-triple calls are tiny
        (~50-200 output tokens each), so claude-3-haiku's 4,096 token output
        limit is never approached.

        Maps the returned _en values back onto each triple so the output
        structure matches the original schema:
            subject.name_en, subject.attributes_en
            predicate_en
            object.name_en, object.attributes_en
            relationship_attributes_en
            properties.evidence_quote_en

        Args:
            triples: List of triple dicts (Thai names, no _en fields yet).

        Returns:
            The same triples list with _en fields populated on every triple.
        """
        if not triples:
            return triples

        def _sanitize(text: str) -> str:
            """Replace literal newlines/tabs so JSON stays valid."""
            return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

        # Build one item per triple
        items = []
        for idx, triple in enumerate(triples):
            subj  = triple.get("subject", {})
            obj   = triple.get("object",  {})
            props = triple.get("properties", {})

            item: Dict[str, Any] = {
                "id":           idx,
                "subject_th":   _sanitize(subj.get("name", "")),
                "predicate_th": _sanitize(triple.get("predicate", "")),
                "object_th":    _sanitize(obj.get("name", "")),
            }
            if subj.get("attributes"):
                item["attributes_th"] = {k: _sanitize(str(v)) for k, v in subj["attributes"].items()}
            if obj.get("attributes"):
                item["obj_attributes_th"] = {k: _sanitize(str(v)) for k, v in obj["attributes"].items()}
            if triple.get("relationship_attributes"):
                item["rel_attrs_th"] = {k: _sanitize(str(v)) for k, v in triple["relationship_attributes"].items()}
            if props.get("evidence_quote"):
                item["evidence_quote_th"] = _sanitize(props["evidence_quote"])

            items.append(item)

        # Fire all translation calls concurrently — one per triple
        translations: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {
                executor.submit(self._translate_single_triple, item): item["id"]
                for item in items
            }
            for future in as_completed(future_to_id):
                try:
                    result = future.result()
                    if result and "id" in result:
                        translations[int(result["id"])] = result
                except Exception as e:
                    triple_id = future_to_id[future]
                    print(f"Warning: Translation failed for triple id={triple_id}: {e}")

        # Map translations back onto triples
        for idx, triple in enumerate(triples):
            t = translations.get(idx, {})
            if not t:
                continue

            subj  = triple.setdefault("subject", {})
            obj   = triple.setdefault("object",  {})
            props = triple.setdefault("properties", {})

            if t.get("subject_en"):
                subj["name_en"] = t["subject_en"]
            if t.get("attributes_en"):
                subj["attributes_en"] = t["attributes_en"]
            if t.get("predicate_en"):
                triple["predicate_en"] = t["predicate_en"]
            if t.get("object_en"):
                obj["name_en"] = t["object_en"]
            if t.get("obj_attributes_en"):
                obj["attributes_en"] = t["obj_attributes_en"]
            if t.get("rel_attrs_en"):
                triple["relationship_attributes_en"] = t["rel_attrs_en"]
            if t.get("evidence_quote_en"):
                props["evidence_quote_en"] = t["evidence_quote_en"]

        return triples

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
        1. Enforce the minimum 3-line span required by the extraction prompt (E2 fix).
        2. Extract lines from original_lines using start/end (1-based, inclusive).
        3. Join them into evidence_quote.
        4. Remove evidence_lines from properties.
        5. Keep evidence_quote_en as-is (provided by LLM for Thai sources).

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
                end   = int(evidence_lines.get("end", start))

                # E2: enforce minimum span before clamping
                if (end - start + 1) < MIN_SPAN:
                    end = start + MIN_SPAN - 1

                # Clamp to valid range (1-based, inclusive)
                start = max(1, min(start, n_lines))
                end   = max(start, min(end, n_lines))

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

        # Stamp each triple with chunk_id and optional community_id.
        # E3 fix: the redundant top-level all_triples flat list has been removed.
        # The refiner (and all downstream consumers) read from chunks[].triples,
        # so the flat list provided no value and roughly doubled the file size.
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

        output_data = {
            "source_file":   str(output_file),
            "total_chunks":  len(triple_results),
            "total_triples": total_triples,
            "llm_provider":  self.llm_provider,
            "llm_model":     self.llm_model,
            "chunks":        triple_results,
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