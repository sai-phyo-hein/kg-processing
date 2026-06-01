"""Semantic chunking module for markdown documents based on LLM analysis."""

import json
import re
import tiktoken
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.model_setup import CHUNKING_PROVIDER, CHUNKING_MODEL
from kg_extractor.utils.prompts import create_chunking_prompt
from kg_extractor.utils.llm_response_parser import parse_chunk_splits_response


class SemanticChunker:
    """Chunk markdown documents using 3-step LLM-assisted approach.

    Process:
    1. Pre-process: Extract <start>...</end> content, split into 1-sentence-1-line,
       remove non-word sentences, number lines [1], [2], etc.
    2. LLM Analysis: Send numbered text to LLM, get chunk boundaries (start/end line numbers)
    3. Programmatic Chunking: Extract text based on LLM-defined boundaries

    Features:
    - Clean 1-sentence-per-line format for better LLM understanding
    - Minimal LLM output (just line numbers)
    - Exact content extraction without rewrites
    - Comprehensive error handling and validation
    - Graceful fallback to rule-based chunking
    """

    def __init__(
        self,
        chunk_granularity: float = 0.5,
        llm_provider: str = CHUNKING_PROVIDER,
        llm_model: str = CHUNKING_MODEL,
        max_retries: int = 3,
        enable_fallback: bool = True,
    ):
        """Initialize the semantic chunker.

        Args:
            chunk_granularity: Threshold for detecting topic changes (0.0-1.0)
            llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
            llm_model: Model to use for LLM analysis
            max_retries: Maximum number of retry attempts for API calls
            enable_fallback: Enable fallback to simple chunking if LLM fails
        """
        self.chunk_granularity = chunk_granularity
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        self.context_limit = 50000  # Token limit per section
        self.CONTEXT_OVERLAP = 50  # Lines to overlap between sections
        self.SAFETY_MARGIN = 500  # Safety margin for token calculations

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            # Get encoding for the model
            if "gpt-4" in self.llm_model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.llm_model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base encoding (works for most models)
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except Exception as e:
            print(f"Warning: Failed to count tokens: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

    def _extract_content_between_flags(self, content: str) -> str:
        """Extract content between <start> and </end> flags.

        Args:
            content: Text content to extract from

        Returns:
            Content between flags, or entire content if flags not found
        """
        match = re.search(r'<start>(.*?)</end>', content, re.DOTALL)
        if match:
            return match.group(1)
        return content

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences. One sentence per line.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences
        """
        # Simple sentence splitting: . ! ? followed by space or end of string
        # This regex splits on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_valid_sentence(self, sentence: str) -> bool:
        """Check if sentence is valid (contains actual words, not just symbols).

        Args:
            sentence: Sentence to validate

        Returns:
            True if sentence contains meaningful content
        """
        # Remove common non-word characters
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
        # Check if at least 2 words remain
        words = cleaned.split()
        return len(words) >= 2

    def _number_and_clean_sentences(self, sentences: List[str]) -> List[str]:
        """Filter valid sentences and number them as [1], [2], etc.

        Args:
            sentences: List of sentences to process

        Returns:
            List of numbered valid sentences
        """
        numbered = []
        line_num = 1
        for sentence in sentences:
            if self._is_valid_sentence(sentence):
                numbered.append(f"[{line_num}] {sentence}")
                line_num += 1
        return numbered

    def chunk_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a markdown file using 3-step LLM-assisted approach.

        Args:
            file_path: Path to the markdown file to chunk

        Returns:
            List of chunk dictionaries with chunk_id and content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If document is invalid
            RuntimeError: If processing fails
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the file content
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

        # Step 1: Pre-process content
        print("📋 Step 1: Pre-processing content...")
        
        # Extract content between flags
        extracted = self._extract_content_between_flags(content)
        
        # Split into sentences
        sentences = self._split_into_sentences(extracted)
        print(f"  📝 Split into {len(sentences)} raw sentences")
        
        # Number and clean sentences
        numbered_lines = self._number_and_clean_sentences(sentences)
        print(f"  ✅ Cleaned to {len(numbered_lines)} valid sentences")
        
        if not numbered_lines:
            raise ValueError("No valid sentences found in document")

        # Step 2: Get chunk boundaries from LLM
        print("🧠 Step 2: Analyzing with LLM...")
        numbered_text = "\n".join(numbered_lines)
        chunk_boundaries = self._get_chunk_boundaries_from_llm(numbered_text, file_path)
        
        if not chunk_boundaries:
            print("⚠️  LLM returned no chunk boundaries, using fallback...")
            if self.enable_fallback:
                return self._fallback_chunking_from_numbered(numbered_lines)
            else:
                raise RuntimeError("Failed to get chunk boundaries from LLM")

        # Step 3: Extract chunks programmatically
        print("📦 Step 3: Extracting chunks...")
        chunks = self._extract_chunks_from_boundaries(
            numbered_lines, chunk_boundaries
        )
        
        print(f"✅ Successfully created {len(chunks)} chunks")
        return chunks

    def _get_chunk_boundaries_from_llm(
        self, numbered_text: str, file_path: str
    ) -> List[Dict[str, int]]:
        """Send numbered text to LLM and get chunk boundaries.
        
        Automatically splits large texts into sections if they exceed token limit.

        Args:
            numbered_text: Text with [N] numbered lines
            file_path: Path to source file for context

        Returns:
            List of chunk boundaries: [{"start": 1, "end": 10}, ...]
        """
        # Check if text is too large for a single LLM call
        token_count = self._count_tokens(numbered_text)
        max_tokens_for_llm = 8000  # Conservative limit for single call
        
        print(f"  📊 Token count: {token_count}")
        
        if token_count > max_tokens_for_llm:
            print(f"  ⚠️  Text too large ({token_count} tokens), splitting into sections...")
            return self._split_and_analyze_sections(numbered_text, file_path)
        else:
            # Single LLM call for smaller texts
            prompt = self._create_boundary_prompt(numbered_text, file_path)
            try:
                response = self._get_llm_response_with_retry(prompt)
                boundaries = self._parse_chunk_boundaries(response)
                if boundaries:
                    print(f"  ✅ Got {len(boundaries)} chunk boundaries from LLM")
                return boundaries
            except Exception as e:
                print(f"⚠️  Failed to get LLM response: {e}")
                return []

    def _split_and_analyze_sections(
        self, numbered_text: str, file_path: str
    ) -> List[Dict[str, int]]:
        """Split large numbered text into sections and analyze each separately.

        Args:
            numbered_text: Full numbered text (possibly very large)
            file_path: Path to source file

        Returns:
            Combined chunk boundaries from all sections
        """
        lines = numbered_text.split("\n")
        max_lines_per_section = 100  # Process ~100 lines per LLM call
        
        all_boundaries = []
        section_num = 1
        
        for i in range(0, len(lines), max_lines_per_section):
            section_lines = lines[i:i + max_lines_per_section]
            section_text = "\n".join(section_lines)
            
            # Extract actual line numbers from the section
            section_line_numbers = []
            for line in section_lines:
                if line.startswith("["):
                    # Extract number from [N] format
                    match = re.match(r'\[(\d+)\]', line)
                    if match:
                        section_line_numbers.append(int(match.group(1)))
            
            if not section_line_numbers:
                continue
            
            section_start = section_line_numbers[0]
            section_end = section_line_numbers[-1]
            
            print(f"  📖 Analyzing section {section_num} (lines {section_start}-{section_end})...")
            
            # Create prompt for this section
            prompt = self._create_boundary_prompt(section_text, file_path)
            
            try:
                response = self._get_llm_response_with_retry(prompt)
                boundaries = self._parse_chunk_boundaries(response)
                
                if boundaries:
                    print(f"    ✅ Got {len(boundaries)} boundaries from section {section_num}")
                    all_boundaries.extend(boundaries)
                    
            except Exception as e:
                print(f"    ⚠️  Section {section_num} failed: {e}")
                continue
            
            section_num += 1
        
        if all_boundaries:
            print(f"  ✅ Combined: {len(all_boundaries)} total boundaries from {section_num - 1} sections")
        
        return all_boundaries

    def _create_boundary_prompt(self, numbered_text: str, file_path: str) -> str:
        """Create prompt for LLM to define chunk boundaries.

        Args:
            numbered_text: Numbered text lines
            file_path: Source file path

        Returns:
            Prompt string
        """
        prompt = f"""Analyze the following numbered lines and define logical chunks.

**TEXT:**
{numbered_text}

**TASK:**
1. Identify logical groupings of lines that form coherent chunks
2. Return chunk boundaries as JSON with "start" and "end" line numbers (inclusive)
3. Do NOT rewrite or modify the text, only define boundaries

**RESPONSE FORMAT:**
Return a JSON object like:
{{
  "chunks": [
    {{"chunk_id": 1, "start": 1, "end": 5}},
    {{"chunk_id": 2, "start": 6, "end": 12}},
    {{"chunk_id": 3, "start": 13, "end": 20}}
  ]
}}

Return ONLY valid JSON, no markdown code blocks or extra text."""
        
        return prompt

    def _get_llm_response_with_retry(self, prompt: str) -> str:
        """Get LLM response with retry logic.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text

        Raises:
            RuntimeError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self._get_llm_response(prompt)
                return response
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")

        return ""

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        try:
            import os
            from langchain_openai import ChatOpenAI
            from kg_extractor.utils.parser import (
                get_api_key,
                get_groq_api_key,
                get_openrouter_api_key,
            )

            # Create LLM based on provider
            if self.llm_provider == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=openai_api_key,
                )
            elif self.llm_provider == "groq":
                groq_api_key = get_groq_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            elif self.llm_provider == "nvidia":
                nvidia_api_key = get_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=nvidia_api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                )
            else:  # openrouter
                openrouter_api_key = get_openrouter_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                )

            # Get response
            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            # Fallback: return empty response
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"chunks": []}'

    def _parse_chunk_boundaries(self, response: str) -> List[Dict[str, int]]:
        """Parse LLM response to extract chunk boundaries.

        Args:
            response: LLM response text

        Returns:
            List of chunk boundaries: [{"chunk_id": 1, "start": 1, "end": 5}, ...]
        """
        try:
            data = parse_json_with_repair(response)
            if not data:
                print(f"Warning: Failed to parse JSON from response")
                return []
            
            if "chunks" not in data or not isinstance(data["chunks"], list):
                print(f"Warning: Invalid response structure: {list(data.keys())}")
                return []
            
            chunks = data["chunks"]
            
            # Validate each chunk has start and end
            valid_chunks = []
            for chunk in chunks:
                if isinstance(chunk, dict) and "start" in chunk and "end" in chunk:
                    valid_chunks.append({
                        "chunk_id": chunk.get("chunk_id", len(valid_chunks) + 1),
                        "start": int(chunk["start"]),
                        "end": int(chunk["end"]),
                    })
            
            return sorted(valid_chunks, key=lambda x: x["start"])
        
        except Exception as e:
            print(f"Warning: Failed to parse chunk boundaries: {e}")
            return []

    def _extract_chunks_from_boundaries(
        self, numbered_lines: List[str], boundaries: List[Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """Extract chunks based on LLM-defined boundaries.

        Args:
            numbered_lines: List of numbered text lines
            boundaries: List of chunk boundaries with start/end line numbers

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        for boundary in boundaries:
            start = boundary["start"]
            end = boundary["end"]
            chunk_id = boundary.get("chunk_id", len(chunks) + 1)
            
            # Extract lines for this chunk (convert 1-based to 0-based indexing)
            chunk_lines = numbered_lines[start - 1:end]
            
            # Remove line numbers and join content
            content = " ".join(line.split("] ", 1)[1] if "] " in line else line 
                              for line in chunk_lines)
            
            # Normalize content
            content = self._normalize_content(content.strip())
            
            if content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                })
        
        return chunks

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Normalize chunk content."""
        return content.replace("\n", " ").replace('"', "'").strip()

    def _fallback_chunking_from_numbered(
        self, numbered_lines: List[str]
    ) -> List[Dict[str, Any]]:
        """Fallback chunking using fixed line groups.

        Args:
            numbered_lines: List of numbered lines

        Returns:
            List of chunk dictionaries
        """
        print("📋 Using simple fallback: 5 lines per chunk...")
        
        chunks = []
        lines_per_chunk = 5
        chunk_id = 1
        
        for i in range(0, len(numbered_lines), lines_per_chunk):
            chunk_lines = numbered_lines[i:i + lines_per_chunk]
            content = " ".join(line.split("] ", 1)[1] if "] " in line else line 
                              for line in chunk_lines)
            content = self._normalize_content(content.strip())
            
            if content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                })
                chunk_id += 1
        
        print(f"✅ Created {len(chunks)} chunks using fallback method")
        return chunks

    def chunk_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a markdown file using sliding window approach with context continuity.

        Processes large documents in sections that fit within context window,
        maintaining continuity by remembering where each section left off.

        Args:
            file_path: Path to the markdown file to chunk

        Returns:
            List of chunk dictionaries with chunk_id and content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If document is invalid
            RuntimeError: If processing fails after retries
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the file content
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

        # Extract content between <start> and </end> flags
        extracted_content = self._extract_content_between_flags(content)
        if not extracted_content:
            extracted_content = content
        
        # Convert extracted content back to lines for processing
        lines = extracted_content.split("\n")
        # Add newlines back to match original format
        lines = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]

        # Use sliding window approach for large documents
        print(f"🧠 Analyzing document with sliding window (context limit: {self.context_limit} tokens per section)...")
        chunk_boundaries = self._analyze_with_sliding_window(lines, file_path)

        # Validate LLM response
        if not chunk_boundaries:
            print("⚠️  Sliding window returned empty chunks, using fallback...")
            if self.enable_fallback:
                return self._fallback_chunking(content, file_path)
            else:
                raise RuntimeError("Sliding window failed to produce valid chunks")

        # _analyze_with_sliding_window now returns fully formed chunks directly
        chunks = chunk_boundaries

        print(f"✅ Successfully created {len(chunks)} chunks")
        return chunks

    def _analyze_with_sliding_window(
        self, lines: List[str], file_path: str
    ) -> List[Dict[str, int]]:
        """Analyze document using sliding window approach with context continuity.

        The LLM is shown numbered lines and asked only for the line numbers where
        new chunks should begin (boundary detection). Actual text is extracted
        programmatically, ensuring 100% content coverage with minimal LLM output.

        Args:
            lines: List of lines from the markdown file
            file_path: Path to the file (for context)

        Returns:
            List of chunk boundary dictionaries with content

        Raises:
            RuntimeError: If processing fails after retries
        """
        all_chunks = []
        current_line = 0
        total_lines = len(lines)
        section_number = 1
        chunk_id_counter = 1

        # Overlap lines shown to LLM for context, but boundaries only requested
        # for the non-overlapping "new" portion to avoid duplicate chunks.
        context_overlap = self.CONTEXT_OVERLAP

        while current_line < total_lines:
            # Determine section size
            section_lines = self._calculate_section_size(
                lines[current_line:],
                context_overlap
            )
            section_end = current_line + section_lines

            print(f"📖 Processing section {section_number} "
                  f"(lines {current_line + 1}-{section_end})...")

            section_content_lines = lines[current_line:section_end]

            # Build numbered-line content for the prompt
            numbered_content = self._build_numbered_content(
                section_content_lines, start_number=1
            )

            # Create prompt requesting only split-point line numbers
            prompt = self._create_chunking_prompt_with_context(
                numbered_content,
                file_path,
                section_number,
                chunk_id_counter,
            )

            try:
                response = self._get_llm_response_with_retry(prompt)
                # Returns list of relative line numbers (1-based within this section)
                split_points = self._parse_llm_response(response, len(section_content_lines))

                if not split_points:
                    print(f"⚠️  Section {section_number} returned no split points, "
                          "treating section as one chunk...")
                    split_points = [1]

                # Enforce minimum chunk size: merge any split that is too close to the previous
                MIN_LINES_PER_CHUNK = 3
                if len(split_points) > 1:
                    filtered = [split_points[0]]
                    for sp in split_points[1:]:
                        if sp - filtered[-1] >= MIN_LINES_PER_CHUNK:
                            filtered.append(sp)
                    if len(filtered) < len(split_points):
                        print(f"  ℹ️  Merged over-split points: {len(split_points)} → {len(filtered)} splits")
                    split_points = filtered

                # Convert relative 1-based line numbers to absolute 0-based indices
                # and build chunks by slicing the lines array
                abs_splits = [current_line + (sp - 1) for sp in split_points]
                # Append sentinel for slicing the last chunk
                abs_splits.append(section_end)

                section_chunks = []
                for i in range(len(abs_splits) - 1):
                    start = abs_splits[i]
                    end = abs_splits[i + 1]
                    content = self._normalize_content("".join(lines[start:end]).strip())
                    if content:
                        section_chunks.append({
                            "chunk_id": chunk_id_counter,
                            "content": content,
                        })
                        chunk_id_counter += 1

                all_chunks.extend(section_chunks)
                print(f"✅ Section {section_number} produced {len(section_chunks)} chunks")

            except Exception as e:
                print(f"⚠️  Section {section_number} failed: {e}")
                if self.enable_fallback:
                    fallback_chunks = self._fallback_chunk_section(
                        section_content_lines,
                        base_chunk_id=chunk_id_counter,
                    )
                    all_chunks.extend(fallback_chunks)
                    chunk_id_counter += len(fallback_chunks)
                    print(f"🔄 Used fallback for section {section_number}")
                else:
                    raise RuntimeError(f"Failed to process section {section_number}: {e}")

            # Advance by the non-overlapping portion so next section has fresh content.
            # The overlap lines were shown for context but their chunks are already recorded.
            advance = max(section_lines - context_overlap, 1)
            current_line += advance
            section_number += 1

        return all_chunks

    def _build_numbered_content(self, lines: List[str], start_number: int = 1) -> str:
        """Build a numbered-line representation for the LLM prompt.

        Args:
            lines: Lines to number
            start_number: Starting line number (1-based)

        Returns:
            String with each line prefixed by its line number
        """
        numbered = []
        for i, line in enumerate(lines):
            num = start_number + i
            numbered.append(f"[{num:04d}] {line}" if line.endswith("\n") else f"[{num:04d}] {line}\n")
        return "".join(numbered)

    def _calculate_section_size(self, lines: List[str], context_overlap: int) -> int:
        """Calculate optimal section size based on token limit.

        Args:
            lines: Available lines for this section
            context_overlap: Number of lines to overlap with next section

        Returns:
            Number of lines to include in this section
        """
        # Start with all available lines
        section_lines = len(lines)

        # Maximum section size to ensure good chunking granularity
        # Even if document fits in context, we want multiple sections for better chunking
        max_section_lines = 1000  # Process at most 1000 lines per section

        # If we have more lines than the max, limit to max
        if section_lines > max_section_lines:
            section_lines = max_section_lines

        # Calculate tokens for progressively smaller sections
        while section_lines > 100:  # Minimum section size
            test_content = "".join(lines[:section_lines])
            token_count = self._count_tokens(test_content)

            # Account for prompt overhead (roughly 1000 tokens for instructions)
            prompt_overhead = 1000
            total_tokens = token_count + prompt_overhead

            if total_tokens <= self.context_limit:
                # This section fits, add some buffer for context overlap
                return min(section_lines + context_overlap, len(lines))

            # Section too large, reduce by 20%
            section_lines = int(section_lines * 0.8)

        # If we get here, use minimum size
        return max(section_lines, 100)

    def _create_chunking_prompt_with_context(
        self,
        numbered_content: str,
        file_path: str,
        section_number: int,
        start_chunk_id: int,
    ) -> str:
        """Create chunking prompt showing numbered lines and requesting split points.

        Args:
            numbered_content: Section content with each line prefixed by [NNNN]
            file_path: Path to the file
            section_number: Current section number
            start_chunk_id: Starting chunk ID for this section (informational)

        Returns:
            Prompt string for the LLM
        """
        base_prompt = create_chunking_prompt(
            content=numbered_content,
            file_path=file_path,
            chunk_granularity=self.chunk_granularity,
        )

        enhanced_prompt = f"""{base_prompt}

**SECTION CONTEXT:**
- This is section {section_number} of a larger document
- Chunks will be numbered starting from {start_chunk_id}
- Return ONLY line numbers (the [NNNN] values) where new chunks should begin
"""
        return enhanced_prompt

    def _get_llm_response_with_retry(self, prompt: str) -> str:
        """Get LLM response with retry logic.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text

        Raises:
            RuntimeError: If all retry attempts fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self._get_llm_response(prompt)
                return response
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}")

        return ""

    @staticmethod
    def _normalize_content(content: str) -> str:
        """Normalize chunk content: collapse newlines to spaces, replace double quotes with single quotes."""
        return content.replace("\n", " ").replace('"', "'")

    def _fallback_chunking(
        self, content: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Fallback to simple rule-based chunking for entire document.

        Args:
            content: Document content
            file_path: Path to the file

        Returns:
            List of chunk dictionaries
        """
        print("📋 Using simple rule-based chunking for entire document...")

        # Split by headers
        chunks = []
        lines = content.split("\n")
        current_chunk = []
        chunk_id = 1

        for line in lines:
            # Check for header
            if line.startswith("#") and current_chunk:
                # Save current chunk
                chunk_content = self._normalize_content("\n".join(current_chunk).strip())
                if chunk_content:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": chunk_content,
                        "topic": f"Section {chunk_id}",
                    })
                    chunk_id += 1
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = self._normalize_content("\n".join(current_chunk).strip())
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "topic": f"Section {chunk_id}",
                })

        print(f"✅ Created {len(chunks)} chunks using fallback method")
        return chunks

    def _fallback_chunk_section(
        self, lines: List[str], base_chunk_id: int
    ) -> List[Dict[str, Any]]:
        """Fallback chunking for a single section.

        Args:
            lines: Lines in the section
            base_chunk_id: Starting chunk ID

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_chunk = []
        chunk_id = base_chunk_id

        for line in lines:
            # Check for header
            if line.startswith("#") and current_chunk:
                # Save current chunk
                chunk_content = self._normalize_content("\n".join(current_chunk).strip())
                if chunk_content:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "content": chunk_content,
                        "topic": f"Section {chunk_id}",
                    })
                    chunk_id += 1
                current_chunk = [line]
            else:
                current_chunk.append(line)

        # Don't forget the last chunk
        if current_chunk:
            chunk_content = self._normalize_content("\n".join(current_chunk).strip())
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "topic": f"Section {chunk_id}",
                })

        return chunks

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        try:
            from kg_extractor.utils.model_setup import get_llm_response
            return get_llm_response(prompt, self.llm_provider, self.llm_model, temperature=0.3)
        except Exception as e:
            # Fallback: return empty response
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"chunks": []}'

    def _parse_llm_response(self, response: str, section_line_count: int) -> List[int]:
        """Parse LLM response to extract chunk split-point line numbers.

        Args:
            response: LLM response text
            section_line_count: Total lines in the section (for validation)

        Returns:
            Sorted list of 1-based relative line numbers where new chunks begin.
            Always starts with 1.
        """
        return parse_chunk_splits_response(response, section_line_count)

    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str) -> str:
        """Save chunks to a JSON file with comprehensive metadata.

        Args:
            chunks: List of chunk dictionaries with chunk_id and content
            output_path: Path to save the chunks

        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Calculate chunk statistics
        chunk_sizes = [len(chunk.get("content", "")) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0

        # Create output structure with comprehensive metadata
        output_data = {
            "source_file": str(output_file),
            "total_chunks": len(chunks),
            "processing_metadata": {
                "method": "sliding_window",
                "chunk_granularity": self.chunk_granularity,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
                "context_limit": self.context_limit,
                "context_overlap": self.CONTEXT_OVERLAP,
                "max_retries": self.max_retries,
                "fallback_enabled": self.enable_fallback,
                "safety_margin": self.SAFETY_MARGIN,
            },
            "chunk_statistics": {
                "total_characters": sum(chunk_sizes),
                "average_chunk_size": round(avg_size, 2),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            },
            "chunks": chunks,
        }

        # Save to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved chunks to: {output_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to save chunks: {e}")

        return str(output_file)


def chunk_markdown_file(
    file_path: str,
    chunk_granularity: float = 0.5,
    llm_provider: str = CHUNKING_PROVIDER,
    llm_model: str = CHUNKING_MODEL,
    output_dir: str = None,
    max_retries: int = 3,
    enable_fallback: bool = True,
) -> str:
    """Chunk a markdown file using LLM analysis with comprehensive error handling.

    Args:
        file_path: Path to the markdown file to chunk
        chunk_granularity: Threshold for detecting topic changes (0.0-1.0)
        llm_provider: LLM provider to use
        llm_model: Model to use for LLM analysis
        output_dir: Directory to save chunks (default: project root/output)
        max_retries: Maximum number of retry attempts for API calls
        enable_fallback: Enable fallback to simple chunking if LLM fails

    Returns:
        Path to saved chunks file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If document is too large and fallback is disabled
        RuntimeError: If processing fails after all retries
    """
    # Set default output directory to project root/output
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent.parent.parent / "output")

    # Create chunker with enhanced configuration
    chunker = SemanticChunker(
        chunk_granularity=chunk_granularity,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_retries=max_retries,
        enable_fallback=enable_fallback,
    )

    # Chunk the file with comprehensive error handling
    try:
        chunks = chunker.chunk_markdown(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to chunk document: {e}")

    # Generate output path
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_chunks.json"
    output_path = Path(output_dir) / output_filename

    # Save chunks
    try:
        result_path = chunker.save_chunks(chunks, str(output_path))
    except Exception as e:
        raise RuntimeError(f"Failed to save chunks: {e}")

    return result_path
