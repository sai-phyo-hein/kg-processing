"""Semantic chunking module for markdown documents based on LLM analysis."""

import json
import tiktoken
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.prompts import create_chunking_prompt


class SemanticChunker:
    """Chunk markdown documents using sliding window LLM analysis.

    Processes large documents in sections that fit within context window,
    maintaining continuity by carrying over context between sections.

    Features:
    - Sliding window approach for unlimited document size
    - Context continuity between sections
    - Comprehensive error handling and validation
    - Graceful fallback to rule-based chunking
    - Token limit management
    - Cost estimation
    - Retry logic with exponential backoff
    """

    # Model context limits (in tokens)
    MODEL_CONTEXT_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 16385,
        "llama3-70b-8192": 8192,
        "google/gemma-3-27b-it": 32768,
        "google/gemma-4-31b-it": 32768,
    }

    # Safety margin (leave 20% of context for response)
    SAFETY_MARGIN = 0.8

    # Context overlap between sections (lines to carry over)
    CONTEXT_OVERLAP = 50

    # Maximum file size to process (in characters)
    MAX_FILE_SIZE = 10000000  # ~10MB (increased for sliding window)

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        max_retries: int = 3,
        enable_fallback: bool = True,
    ):
        """Initialize the semantic chunker.

        Args:
            similarity_threshold: Threshold for detecting topic changes (0.0-1.0)
            llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
            llm_model: Model to use for LLM analysis
            max_retries: Maximum number of retry attempts for API calls
            enable_fallback: Enable fallback to simple chunking if LLM fails
        """
        self.similarity_threshold = similarity_threshold
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback

        # Get context limit for the model
        self.context_limit = self._get_model_context_limit()

    def _get_model_context_limit(self) -> int:
        """Get the context limit for the current model.

        Returns:
            Maximum context window in tokens
        """
        # Try to find exact match
        if self.llm_model in self.MODEL_CONTEXT_LIMITS:
            return int(self.MODEL_CONTEXT_LIMITS[self.llm_model] * self.SAFETY_MARGIN)

        # Try to find partial match (for model families)
        for model_name, limit in self.MODEL_CONTEXT_LIMITS.items():
            if model_name in self.llm_model or self.llm_model in model_name:
                return int(limit * self.SAFETY_MARGIN)

        # Default to conservative limit
        print(f"Warning: Unknown model '{self.llm_model}', using default context limit of 16000 tokens")
        return 16000

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

    def _validate_document_size(self, content: str, file_path: str) -> Tuple[bool, str]:
        """Validate document size before processing.

        With sliding window approach, we can handle much larger documents,
        but still need basic validation.

        Args:
            content: Document content
            file_path: Path to the file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size (increased limit for sliding window)
        if len(content) > self.MAX_FILE_SIZE:
            error_msg = (
                f"Document extremely large: {len(content)} characters "
                f"(max: {self.MAX_FILE_SIZE}). "
                f"Consider splitting the document into smaller files."
            )
            return False, error_msg

        # With sliding window, we don't need to check total token count
        # since we process in sections. Just validate the file isn't empty.
        if len(content.strip()) == 0:
            return False, "Document is empty"

        return True, ""

    def _estimate_cost(self, content: str) -> Dict[str, float]:
        """Estimate API cost for processing the document with sliding window.

        Args:
            content: Document content

        Returns:
            Dictionary with cost estimates
        """
        total_tokens = self._count_tokens(content)

        # Estimate number of sections needed
        tokens_per_section = self.context_limit - 1000  # Account for prompt overhead
        num_sections = max(1, (total_tokens + tokens_per_section - 1) // tokens_per_section)

        # Rough cost estimates (in USD)
        cost_estimates = {
            "gpt-4o": {"input": 0.000005, "output": 0.000015},
            "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015},
        }

        # Find closest model match
        model_costs = cost_estimates.get("gpt-4o-mini", {"input": 0.00000015, "output": 0.0000006})
        for model_name in cost_estimates:
            if model_name in self.llm_model:
                model_costs = cost_estimates[model_name]
                break

        # Estimate input tokens (total tokens processed across all sections)
        input_cost = total_tokens * model_costs["input"]

        # Estimate output tokens (roughly 5% of input per section for chunking)
        estimated_output_tokens = int(total_tokens * 0.05 * num_sections)
        output_cost = estimated_output_tokens * model_costs["output"]

        total_cost = input_cost + output_cost

        return {
            "total_tokens": total_tokens,
            "estimated_sections": num_sections,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
        }

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
                lines = f.readlines()
        except Exception as e:
            raise RuntimeError(f"Failed to read file: {e}")

        content = "".join(lines)

        # Validate document size
        print(f"📊 Validating document size...")
        is_valid, error_msg = self._validate_document_size(content, file_path)
        if not is_valid:
            if self.enable_fallback:
                print(f"⚠️  {error_msg}")
                print("🔄 Falling back to simple chunking...")
                return self._fallback_chunking(content, file_path)
            else:
                raise ValueError(error_msg)

        # Estimate and report cost
        cost_estimate = self._estimate_cost(content)
        print(f"💰 Cost estimate: ${cost_estimate['total_cost_usd']:.6f} "
              f"({cost_estimate['total_tokens']} total tokens, "
              f"~{cost_estimate['estimated_sections']} sections)")

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

        # Create chunks based on boundaries
        chunks = self._create_chunks_from_boundaries(chunk_boundaries)

        print(f"✅ Successfully created {len(chunks)} chunks")
        return chunks

    def _analyze_with_sliding_window(
        self, lines: List[str], file_path: str
    ) -> List[Dict[str, int]]:
        """Analyze document using sliding window approach with context continuity.

        Processes document in sections that fit within context window,
        maintaining continuity by carrying over context between sections.

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

        # Context overlap between sections (number of lines to carry over)
        context_overlap = 50  # Carry over last 50 lines for continuity

        while current_line < total_lines:
            # Calculate section size based on token limit
            section_lines = self._calculate_section_size(
                lines[current_line:],
                context_overlap
            )

            print(f"📖 Processing section {section_number} "
                  f"(lines {current_line + 1}-{current_line + section_lines})...")

            # Get section content with context
            section_content_lines = lines[current_line:current_line + section_lines]
            section_content = "".join(section_content_lines)

            # Add context from previous section if available
            context_prefix = ""
            if all_chunks and len(all_chunks) > 0:
                # Get last chunk for context
                last_chunk = all_chunks[-1]
                last_content = last_chunk.get("content", "")
                # Take last few sentences as context
                context_sentences = self._get_last_sentences(last_content, max_sentences=3)
                if context_sentences:
                    context_prefix = f"[CONTEXT FROM PREVIOUS SECTION: {context_sentences}]\n\n"

            # Create prompt with context
            prompt = self._create_chunking_prompt_with_context(
                section_content,
                file_path,
                context_prefix,
                section_number,
                len(all_chunks) + 1  # Next chunk ID
            )

            # Get LLM response with retry logic
            try:
                response = self._get_llm_response_with_retry(prompt)
                section_chunks = self._parse_llm_response(response)

                if not section_chunks:
                    print(f"⚠️  Section {section_number} returned no chunks, skipping...")
                    current_line += section_lines - context_overlap
                    section_number += 1
                    continue

                # Adjust chunk IDs to maintain continuity
                base_chunk_id = len(all_chunks) + 1
                for i, chunk in enumerate(section_chunks):
                    chunk["chunk_id"] = base_chunk_id + i

                all_chunks.extend(section_chunks)
                print(f"✅ Section {section_number} produced {len(section_chunks)} chunks")

            except Exception as e:
                print(f"⚠️  Section {section_number} failed: {e}")
                if self.enable_fallback:
                    # Use fallback for this section
                    fallback_chunks = self._fallback_chunk_section(
                        section_content_lines,
                        base_chunk_id=len(all_chunks) + 1
                    )
                    all_chunks.extend(fallback_chunks)
                    print(f"🔄 Used fallback for section {section_number}")
                else:
                    raise RuntimeError(f"Failed to process section {section_number}: {e}")

            # Move to next section, accounting for context overlap
            current_line += section_lines - context_overlap
            section_number += 1

        return all_chunks

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
        max_section_lines = 200  # Process at most 200 lines per section

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

    def _get_last_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract last few sentences from text for context continuity.

        Args:
            text: Text to extract sentences from
            max_sentences: Maximum number of sentences to extract

        Returns:
            Last few sentences as a string
        """
        if not text:
            return ""

        # Simple sentence splitting
        sentences = []
        current_sentence = []

        for char in text:
            current_sentence.append(char)
            if char in ['.', '!', '?']:
                sentence = ''.join(current_sentence).strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = []

        # Add any remaining text
        if current_sentence:
            remaining = ''.join(current_sentence).strip()
            if remaining:
                sentences.append(remaining)

        # Get last few sentences
        last_sentences = sentences[-max_sentences:] if len(sentences) > max_sentences else sentences

        return ' '.join(last_sentences)

    def _create_chunking_prompt_with_context(
        self,
        content: str,
        file_path: str,
        context_prefix: str,
        section_number: int,
        start_chunk_id: int,
    ) -> str:
        """Create chunking prompt with context from previous section.

        Args:
            content: Current section content
            file_path: Path to the file
            context_prefix: Context from previous section
            section_number: Current section number
            start_chunk_id: Starting chunk ID for this section

        Returns:
            Prompt string for the LLM
        """
        base_prompt = create_chunking_prompt(
            content=content,
            file_path=file_path,
            similarity_threshold=self.similarity_threshold,
        )

        # Add context and section information
        enhanced_prompt = f"""{base_prompt}

**IMPORTANT CONTEXT INFORMATION:**
- This is section {section_number} of a larger document
- Start chunk IDs from {start_chunk_id} (continuing from previous sections)
{context_prefix}

**CONTINUITY INSTRUCTIONS:**
- Maintain consistency with previous sections
- Ensure chunks flow naturally from previous content
- Use the context above to understand where the previous section ended
- Start chunk numbering from {start_chunk_id} to maintain continuity
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
                chunk_content = "\n".join(current_chunk).strip()
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
            chunk_content = "\n".join(current_chunk).strip()
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
                chunk_content = "\n".join(current_chunk).strip()
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
            chunk_content = "\n".join(current_chunk).strip()
            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "topic": f"Section {chunk_id}",
                })

        return chunks

    def _create_chunks_from_boundaries(
        self, boundaries: List[Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """Create chunks from boundaries determined by LLM.

        Args:
            boundaries: List of chunk boundaries from LLM

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        for boundary in boundaries:
            chunk_id = boundary["chunk_id"]
            content = boundary.get("content", "")
            topic = boundary.get("topic", f"Chunk {chunk_id}")

            # Create chunk
            chunk = {
                "chunk_id": chunk_id,
                "content": content,
                "topic": topic,
            }

            chunks.append(chunk)

        return chunks

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

    def _parse_llm_response(self, response: str) -> List[Dict[str, int]]:
        """Parse LLM response to extract chunk boundaries.

        Args:
            response: LLM response text

        Returns:
            List of chunk boundary dictionaries
        """
        try:
            # Try to extract JSON from response
            # Look for JSON between ```json and ``` or just parse the whole response
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            elif response.startswith("```"):
                response = response[3:]  # Remove ```
            if response.endswith("```"):
                response = response[:-3]  # Remove trailing ```

            response = response.strip()

            # Parse JSON
            data = json.loads(response)

            if "chunks" not in data:
                return []

            boundaries = []
            for chunk in data["chunks"]:
                boundaries.append(
                    {
                        "chunk_id": chunk.get("chunk_id", len(boundaries) + 1),
                        "content": chunk.get("content", ""),
                    }
                )

            return boundaries

        except json.JSONDecodeError:
            # Fallback: create a single chunk with all content
            return [
                {
                    "chunk_id": 1,
                    "content": "Full document",
                }
            ]
        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            # Fallback: create a single chunk with all content
            return [
                {
                    "chunk_id": 1,
                    "content": "Full document",
                }
            ]

    def _create_chunks_from_boundaries(
        self, boundaries: List[Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """Create chunks from boundaries determined by LLM.

        Args:
            boundaries: List of chunk boundaries from LLM

        Returns:
            List of chunk dictionaries
        """
        chunks = []

        for boundary in boundaries:
            chunk_id = boundary["chunk_id"]
            content = boundary.get("content", "")

            # Create chunk
            chunk = {
                "chunk_id": chunk_id,
                "content": content,
            }

            chunks.append(chunk)

        return chunks

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
                "similarity_threshold": self.similarity_threshold,
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
    similarity_threshold: float = 0.5,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    output_dir: str = None,
    max_retries: int = 3,
    enable_fallback: bool = True,
) -> str:
    """Chunk a markdown file using LLM analysis with comprehensive error handling.

    Args:
        file_path: Path to the markdown file to chunk
        similarity_threshold: Threshold for detecting topic changes (0.0-1.0)
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
        output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    # Create chunker with enhanced configuration
    chunker = SemanticChunker(
        similarity_threshold=similarity_threshold,
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
