"""Triplet extraction module for knowledge graph construction."""

import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from kg_extractor.utils.prompts import create_triple_extraction_prompt


class TripleExtractor:
    """Extract knowledge graph triples from text chunks using LLM analysis."""

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize the triple extractor.

        Args:
            llm_provider: LLM provider to use (openai, groq, nvidia, openrouter)
            llm_model: Model to use for LLM analysis
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    def extract_triples_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        source_file: str = "",
        with_schema: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract triples from multiple chunks in parallel.

        Args:
            chunks: List of chunk dictionaries with chunk_id and content
            source_file: Source file path for context
            with_schema: Whether to validate triples against schema.md

        Returns:
            List of triple extraction results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._extract_triples_from_single_chunk,
                    chunk,
                    source_file,
                    with_schema,
                ): chunk["chunk_id"]
                for chunk in chunks
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to extract triples from chunk {chunk_id}: {e}")
                    # Add empty result for failed chunks
                    results.append({
                        "chunk_id": chunk_id,
                        "triples": [],
                        "error": str(e),
                    })

        # Sort results by chunk_id to maintain order
        results.sort(key=lambda x: x["chunk_id"])

        return results

    def _extract_triples_from_single_chunk(
        self,
        chunk: Dict[str, Any],
        source_file: str = "",
        with_schema: bool = False,
    ) -> Dict[str, Any]:
        """Extract triples from a single chunk.

        Args:
            chunk: Chunk dictionary with chunk_id and content
            source_file: Source file path for context
            with_schema: Whether to validate triples against schema.md

        Returns:
            Dictionary with chunk_id, document_metadata, and extracted triples
        """
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        # Create prompt for triple extraction
        prompt = create_triple_extraction_prompt(content, source_file, chunk_id, with_schema)

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response to extract triples and metadata
        parsed_data = self._parse_llm_response(response)

        return {
            "chunk_id": chunk_id,
            "document_metadata": parsed_data.get("document_metadata", {}),
            "triples": parsed_data.get("discovered_triples", []),
        }

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
            return '{"document_metadata": {"reference_date": null, "source_id": null, "chunk_id": null}, "discovered_triples": []}'

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract triples and metadata.

        Args:
            response: LLM response text

        Returns:
            Dictionary with document metadata and triples
        """
        try:
            # Try to extract JSON from response
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

            # Validate new structure
            if "document_metadata" not in data or "discovered_triples" not in data:
                print(f"Warning: Invalid response structure: {list(data.keys())}")
                return {
                    "document_metadata": {"reference_date": None, "source_id": None, "chunk_id": None},
                    "discovered_triples": []
                }

            return data

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            return {
                "document_metadata": {"reference_date": None, "source_id": None, "chunk_id": None},
                "discovered_triples": []
            }
        except Exception as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return {
                "document_metadata": {"reference_date": None, "source_id": None, "chunk_id": None},
                "discovered_triples": []
            }

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
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    output_dir: str = None,
    with_schema: bool = False,
    community_id: str = None,
) -> str:
    """Extract triples from chunks using LLM analysis and save results.

    Args:
        chunks: List of chunk dictionaries with chunk_id and content
        source_file: Source file path for context
        llm_provider: LLM provider to use
        llm_model: Model to use for LLM analysis
        output_dir: Directory to save triples (default: project root/output)
        with_schema: Whether to validate triples against schema.md
        community_id: Optional community ID to add to each triple's properties

    Returns:
        Path to saved triples file
    """
    # Set default output directory to project root/output
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent.parent.parent / "output")
    # Create extractor
    extractor = TripleExtractor(
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Extract triples from chunks
    triple_results = extractor.extract_triples_from_chunks(chunks, source_file, with_schema)

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
