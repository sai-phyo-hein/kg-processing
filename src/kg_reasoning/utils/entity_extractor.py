"""Entity extraction module for user queries."""

import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from kg_reasoning.utils.prompts import get_entity_extraction_prompt

# Load environment variables
load_dotenv()


class EntityExtractor:
    """Extract entities from user queries using hybrid LLM and Qdrant matching.

    Hybrid Matching Approach:
    - Keyword-based matching (70% weight): Provides precision for exact terminology
    - Semantic embedding matching (30% weight): Handles synonyms and context
    - Combined scoring: Balances precision with flexibility
    - LLM refinement: Selects most relevant matches from combined results

    This approach is optimal for knowledge graph querying because:
    1. Knowledge graphs require precise entity/predicate terminology
    2. Users may use synonyms or different phrasings
    3. Natural language queries need semantic understanding
    4. Debugging is easier with clear match type information
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        keyword_weight: float = 0.7,
        semantic_weight: float = 0.3,
    ):
        """Initialize the entity extractor.

        Args:
            llm_provider: LLM provider (openai, groq, nvidia, openrouter)
            llm_model: Model for LLM analysis
            qdrant_url: Qdrant server URL (from env if not provided)
            qdrant_api_key: Qdrant API key (from env if not provided)
            keyword_weight: Weight for keyword matches (0.0-1.0, default: 0.7)
            semantic_weight: Weight for semantic matches (0.0-1.0, default: 0.3)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight

        # Validate weights
        if abs(keyword_weight + semantic_weight - 1.0) > 0.01:
            print(f"Warning: Weights don't sum to 1.0 (keyword={keyword_weight}, semantic={semantic_weight})")

        # Initialize Qdrant client if credentials provided
        if qdrant_url and qdrant_api_key:
            self._init_qdrant_client()
        else:
            self.qdrant_client = None

    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient

            self.qdrant_client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant client: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            import os
            from openai import OpenAI

            openai_api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=openai_api_key)

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Failed to get embedding: {e}")
            # Fallback to random vector
            import random
            return [random.random() for _ in range(1536)]

    def _query_qdrant_collections(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query multiple Qdrant collections for entities and predicates.

        Args:
            query_embedding: Pre-computed embedding vector
            limit: Maximum number of results per collection
            score_threshold: Minimum similarity score for matches

        Returns:
            Dictionary mapping collection names to their matches
        """
        if not self.qdrant_client:
            return {}

        try:
            from qdrant_client.http import models as qdrant_models

            # Map collection names to their vector names
            collection_vector_names = {
                "entity_registry": "entity",
                "predicate_registry": "predicate"
            }

            results = {}

            for collection_name in ["entity_registry", "predicate_registry"]:
                try:
                    # Get the correct vector name for this collection
                    vector_name = collection_vector_names.get(collection_name, "entity")

                    # Search using vector similarity
                    search_results = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_embedding,
                        using=vector_name,
                        limit=limit,
                        with_payload=True,
                        score_threshold=score_threshold,
                    )

                    # Convert to expected format
                    matches = []
                    for result in search_results.points:
                        matches.append({
                            "id": result.id,
                            "score": result.score,
                            "payload": result.payload,
                        })

                    results[collection_name] = matches

                except Exception as e:
                    print(f"Warning: Failed to query collection {collection_name}: {e}")
                    results[collection_name] = []

            return results

        except Exception as e:
            print(f"Warning: Failed to query Qdrant: {e}")
            return {}

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple NLP techniques.

        Args:
            text: Text to extract keywords from

        Returns:
            List of keywords
        """
        # Convert to lowercase
        text = text.lower()

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
            'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        }

        # Split into words and remove stop words
        words = re.findall(r'\b\w+\b', text)
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Extract multi-word phrases (2-3 words)
        phrases = []
        for i in range(len(words) - 1):
            # 2-word phrases
            if words[i] not in stop_words and words[i+1] not in stop_words:
                phrases.append(f"{words[i]} {words[i+1]}")

            # 3-word phrases
            if i < len(words) - 2:
                if (words[i] not in stop_words and
                    words[i+1] not in stop_words and
                    words[i+2] not in stop_words):
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Combine single words and phrases, remove duplicates
        all_keywords = list(set(keywords + phrases))

        # Sort by length (longer phrases first) and relevance
        all_keywords.sort(key=lambda x: (-len(x), x))

        return all_keywords[:20]  # Return top 20 keywords

    def _query_qdrant_keywords(
        self,
        keywords: List[str],
        limit: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query Qdrant collections using keyword matching.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results per keyword

        Returns:
            Dictionary mapping collection names to their matches
        """
        if not self.qdrant_client or not keywords:
            return {}

        try:
            collections = ["entity_registry", "predicate_registry"]
            results = {}

            for collection_name in collections:
                collection_matches = []

                try:
                    # Fetch all points from the collection (since collections are small)
                    # Use scroll to get all points
                    all_points, _ = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=1000,  # Get up to 1000 points (should be enough)
                        with_payload=True,
                    )

                    # Filter points that match any keyword (case-insensitive)
                    keywords_lower = [k.lower() for k in keywords]

                    for point in all_points:
                        point_name = point.payload.get("name", "")
                        point_name_lower = point_name.lower()

                        # Check if any keyword matches the name
                        for keyword, keyword_lower in zip(keywords, keywords_lower):
                            if keyword_lower in point_name_lower:
                                collection_matches.append({
                                    "id": point.id,
                                    "score": 1.0,  # Perfect match for keywords
                                    "payload": point.payload,
                                    "match_type": "keyword",
                                    "matched_keyword": keyword
                                })
                                break  # Only add once per point

                except Exception as e:
                    print(f"Warning: Failed to search keywords in {collection_name}: {e}")
                    continue

                # Remove duplicates and limit results
                seen_ids = set()
                unique_matches = []
                for match in collection_matches:
                    if match["id"] not in seen_ids:
                        unique_matches.append(match)
                        seen_ids.add(match["id"])

                results[collection_name] = unique_matches[:10]  # Limit to top 10

            return results

        except Exception as e:
            print(f"Warning: Failed to query Qdrant with keywords: {e}")
            return {}

    def _combine_matches(
        self,
        keyword_matches: Dict[str, List[Dict[str, Any]]],
        semantic_matches: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Combine keyword and semantic matches with weighted scoring.

        Args:
            keyword_matches: Results from keyword search
            semantic_matches: Results from semantic search

        Returns:
            Combined results with weighted scores
        """
        combined = {}

        for collection_name in ["entity_registry", "predicate_registry"]:
            keyword_results = keyword_matches.get(collection_name, [])
            semantic_results = semantic_matches.get(collection_name, [])

            # Create a map to track combined scores
            combined_map = {}

            # Process keyword matches
            for match in keyword_results:
                match_id = match["id"]
                if match_id not in combined_map:
                    combined_map[match_id] = {
                        "id": match_id,
                        "payload": match["payload"],
                        "keyword_score": 0.0,
                        "semantic_score": 0.0,
                        "match_type": []
                    }
                combined_map[match_id]["keyword_score"] = max(
                    combined_map[match_id]["keyword_score"],
                    match["score"] * self.keyword_weight
                )
                if "keyword" not in combined_map[match_id]["match_type"]:
                    combined_map[match_id]["match_type"].append("keyword")

            # Process semantic matches
            for match in semantic_results:
                match_id = match["id"]
                if match_id not in combined_map:
                    combined_map[match_id] = {
                        "id": match_id,
                        "payload": match["payload"],
                        "keyword_score": 0.0,
                        "semantic_score": 0.0,
                        "match_type": []
                    }
                combined_map[match_id]["semantic_score"] = max(
                    combined_map[match_id]["semantic_score"],
                    match["score"] * self.semantic_weight
                )
                if "semantic" not in combined_map[match_id]["match_type"]:
                    combined_map[match_id]["match_type"].append("semantic")

            # Calculate combined scores and format results
            combined_results = []
            for match_id, match_data in combined_map.items():
                combined_score = match_data["keyword_score"] + match_data["semantic_score"]

                combined_results.append({
                    "id": match_id,
                    "score": combined_score,
                    "payload": match_data["payload"],
                    "keyword_score": match_data["keyword_score"],
                    "semantic_score": match_data["semantic_score"],
                    "match_type": match_data["match_type"]
                })

            # Sort by combined score
            combined_results.sort(key=lambda x: x["score"], reverse=True)

            combined[collection_name] = combined_results[:10]  # Limit to top 10

        return combined

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
                    temperature=0.3,
                    api_key=openai_api_key,
                )
            elif self.llm_provider == "groq":
                from kg_extractor.utils.parser import get_groq_api_key
                groq_api_key = get_groq_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            elif self.llm_provider == "nvidia":
                from kg_extractor.utils.parser import get_api_key
                nvidia_api_key = get_api_key()
                llm = ChatOpenAI(
                    model=self.llm_model,
                    temperature=0.3,
                    api_key=nvidia_api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                )
            else:  # openrouter
                from kg_extractor.utils.parser import get_openrouter_api_key
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
            print(f"Warning: Failed to get LLM response: {e}")
            return '{"entities": [], "relationships": [], "query_intent": "", "query_type": "exploratory"}'

    def extract_entities(self, user_query: str) -> Dict[str, Any]:
        """Extract entities from user query using hybrid Qdrant matching and LLM refinement.

        Args:
            user_query: The user's natural language query

        Returns:
            Dictionary with extracted entities, predicates, and query intent
        """
        # Step 1: Extract keywords from user query
        print("   Extracting keywords from user query...")
        keywords = self._extract_keywords(user_query)
        print(f"   Extracted {len(keywords)} keywords")

        # Step 2: Embed the whole user query
        print("   Embedding user query...")
        query_embedding = self._get_embedding(user_query)

        # Step 3: Query Qdrant with keywords (higher weight)
        print("   Querying Qdrant with keywords...")
        keyword_matches = self._query_qdrant_keywords(
            keywords,
            limit=5,
        )

        # Step 4: Query Qdrant with embedding (lower weight)
        print("   Querying Qdrant with semantic embedding...")
        semantic_matches = self._query_qdrant_collections(
            query_embedding,
            limit=5,
            score_threshold=0.6,  # Lower threshold for semantic matches
        )

        # Step 5: Combine matches with weighted scoring
        print("   Combining keyword and semantic matches...")
        combined_matches = self._combine_matches(keyword_matches, semantic_matches)

        entity_matches = combined_matches.get("entity_registry", [])
        predicate_matches = combined_matches.get("predicate_registry", [])

        print(f"   Combined results: {len(entity_matches)} entity matches and {len(predicate_matches)} predicate matches")

        # Step 6: Create prompt with combined matches for LLM refinement
        prompt = get_entity_extraction_prompt(
            user_query,
            entity_matches=entity_matches,
            predicate_matches=predicate_matches,
            keywords=keywords,
        )

        # Step 7: Get LLM response with context from combined matches
        print("   Refining extraction with LLM...")
        response = self._get_llm_response(prompt)

        # Step 8: Parse response
        result = self._parse_entity_extraction_response(response)

        # Add match information to result
        result["qdrant_entity_matches"] = entity_matches
        result["qdrant_predicate_matches"] = predicate_matches
        result["keywords"] = keywords
        result["keyword_matches"] = keyword_matches
        result["semantic_matches"] = semantic_matches

        return result

    def _parse_entity_extraction_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for entity extraction.

        Args:
            response: LLM response text

        Returns:
            Dictionary with extracted entities and predicates
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
                "entities": data.get("entities", []),
                "predicates": data.get("predicates", []),
                "query_intent": data.get("query_intent", ""),
                "query_type": data.get("query_type", "exploratory"),
            }

        except json.JSONDecodeError:
            # Fallback: return empty extraction
            print("Warning: Failed to parse entity extraction response")
            return {
                "entities": [],
                "predicates": [],
                "query_intent": "",
                "query_type": "exploratory",
            }
        except Exception as e:
            print(f"Warning: Failed to parse entity extraction response: {e}")
            return {
                "entities": [],
                "predicates": [],
                "query_intent": "",
                "query_type": "exploratory",
            }
