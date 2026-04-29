"""Answer synthesis module for generating natural language responses."""

import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from kg_reasoning.utils.prompts import get_answer_synthesis_prompt, get_query_suggestion_prompt

# Load environment variables
load_dotenv()


class AnswerSynthesizer:
    """Synthesize natural language answers from Neo4j results."""

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize the answer synthesizer.

        Args:
            llm_provider: LLM provider (openai, groq, nvidia, openrouter)
            llm_model: Model for LLM analysis
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model

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

            # Create LLM based on provider
            if self.llm_provider == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.1,  # Low temperature for strict adherence to results
                    api_key=openai_api_key,
                )
            elif self.llm_provider == "groq":
                from kg_extractor.utils.parser import get_groq_api_key
                groq_api_key = get_groq_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.1,  # Low temperature for strict adherence to results
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            elif self.llm_provider == "nvidia":
                from kg_extractor.utils.parser import get_api_key
                nvidia_api_key = get_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.1,  # Low temperature for strict adherence to results
                    api_key=nvidia_api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                )
            else:  # openrouter
                from kg_extractor.utils.parser import get_openrouter_api_key
                openrouter_api_key = get_openrouter_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.1,  # Low temperature for strict adherence to results
                    api_key=openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                )

            # Get response
            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            print(f"Warning: Failed to get LLM response: {e}")
            return "I apologize, but I encountered an error while synthesizing the answer. Please try again."

    def synthesize_answer(
        self,
        user_query: str,
        neo4j_results: List[Dict[str, Any]],
        cypher_query: str,
    ) -> str:
        """Synthesize a natural language answer from Neo4j results.

        Args:
            user_query: The original user query
            neo4j_results: Results from Neo4j query
            cypher_query: The Cypher query that was executed

        Returns:
            Natural language answer
        """
        # Create prompt
        prompt = get_answer_synthesis_prompt(user_query, neo4j_results, cypher_query)

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Clean up response
        return self._clean_response(response)

    def _clean_response(self, response: str) -> str:
        """Clean up the LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response
        """
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            if len(lines) > 1:
                response = "\n".join(lines[1:])
            if response.endswith("```"):
                response = response[:-3]
        response = response.strip()

        return response

    def generate_query_suggestions(
        self,
        user_query: str,
        high_connectivity_nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate query suggestions when no entity matches found.

        Args:
            user_query: The original user query
            high_connectivity_nodes: List of high connectivity nodes from Neo4j

        Returns:
            Dictionary with query suggestions
        """
        # Create prompt
        prompt = get_query_suggestion_prompt(user_query, high_connectivity_nodes)

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response
        return self._parse_query_suggestion_response(response)

    def _parse_query_suggestion_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for query suggestions.

        Args:
            response: LLM response text

        Returns:
            Dictionary with query suggestions
        """
        try:
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            data = json.loads(response)

            return {
                "message": data.get("message", "No direct matches found. Here are some alternative queries:"),
                "suggestions": data.get("suggestions", []),
                "topics": data.get("topics", []),
            }

        except json.JSONDecodeError:
            # Fallback: return basic message
            print("Warning: Failed to parse query suggestion response")
            return {
                "message": "No direct matches found for your query. Please try rephrasing your question.",
                "suggestions": [],
                "topics": [],
            }
        except Exception as e:
            print(f"Warning: Failed to parse query suggestion response: {e}")
            return {
                "message": "No direct matches found for your query. Please try rephrasing your question.",
                "suggestions": [],
                "topics": [],
            }
