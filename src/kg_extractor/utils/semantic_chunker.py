"""Semantic chunking module for markdown documents based on LLM analysis."""

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.prompts import create_chunking_prompt


class SemanticChunker:
    """Chunk markdown documents based on LLM analysis of topic shifts."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize the semantic chunker.

        Args:
            similarity_threshold: Threshold for detecting topic changes (0.0-1.0)
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
            llm_model: Model to use for LLM analysis
        """
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def chunk_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a markdown file based on LLM analysis of topic shifts.

        Args:
            file_path: Path to the markdown file to chunk

        Returns:
            List of chunk dictionaries with chunk_id and content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the file content
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Use LLM to analyze and determine chunk boundaries
        chunk_boundaries = self._analyze_with_llm(lines, file_path)

        # Create chunks based on boundaries
        chunks = self._create_chunks_from_boundaries(chunk_boundaries)

        return chunks

    def _analyze_with_llm(self, lines: List[str], file_path: str) -> List[Dict[str, int]]:
        """Use LLM to analyze content and determine chunk boundaries.

        The LLM reads the content and identifies where topic changes occur.

        Args:
            lines: List of lines from the markdown file
            file_path: Path to the file (for context)

        Returns:
            List of chunk boundary dictionaries with content
        """
        # Prepare content for LLM analysis
        content = "".join(lines)

        # Create prompt for LLM
        prompt = self._create_chunking_prompt(content, file_path)

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response to extract chunk boundaries
        boundaries = self._parse_llm_response(response)

        return boundaries

    def _create_chunking_prompt(self, content: str, file_path: str) -> str:
        """Create a prompt for the LLM to analyze content and determine chunk boundaries.

        Args:
            content: Full content of the markdown file
            file_path: Path to the file

        Returns:
            Prompt string for the LLM
        """
        return create_chunking_prompt(
            content=content,
            file_path=file_path,
            similarity_threshold=self.similarity_threshold,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        )

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
        """Save chunks to a JSON file.

        Args:
            chunks: List of chunk dictionaries with chunk_id and content
            output_path: Path to save the chunks

        Returns:
            Path to saved file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create output structure
        output_data = {
            "source_file": str(output_file),
            "total_chunks": len(chunks),
            "similarity_threshold": self.similarity_threshold,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "chunks": chunks,
        }

        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return str(output_file)


def chunk_markdown_file(
    file_path: str,
    similarity_threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 1000,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    output_dir: str = None,
) -> str:
    """Chunk a markdown file using LLM analysis and save results.

    Args:
        file_path: Path to the markdown file to chunk
        similarity_threshold: Threshold for detecting topic changes (0.0-1.0)
        min_chunk_size: Minimum tokens per chunk
        max_chunk_size: Maximum tokens per chunk
        llm_provider: LLM provider to use
        llm_model: Model to use for LLM analysis
        output_dir: Directory to save chunks (default: project root/output)

    Returns:
        Path to saved chunks file
    """
    # Set default output directory to project root/output
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent / "output")

    # Create chunker
    chunker = SemanticChunker(
        similarity_threshold=similarity_threshold,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Chunk the file
    chunks = chunker.chunk_markdown(file_path)

    # Generate output path
    input_path = Path(file_path)
    output_filename = f"{input_path.stem}_chunks.json"
    output_path = Path(output_dir) / output_filename

    # Save chunks
    result_path = chunker.save_chunks(chunks, str(output_path))

    return result_path
