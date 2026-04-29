"""Cypher query generation module for Neo4j."""

import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from kg_reasoning.utils.prompts import get_cypher_generation_prompt, get_high_connectivity_query_prompt

# Load environment variables
load_dotenv()


class CypherGenerator:
    """Generate Cypher queries from natural language queries."""

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
    ):
        """Initialize the Cypher generator.

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
            return '{"cypher_query": "", "query_explanation": "LLM error", "expected_result_type": "unknown", "result_structure": ""}'

    def generate_cypher(
        self,
        refined_query: str,
        schema_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher query from refined natural language query.

        Args:
            refined_query: The refined user query with canonical entities
            schema_info: Optional information about the graph schema

        Returns:
            Dictionary with Cypher query and metadata
        """
        # Create prompt
        prompt = get_cypher_generation_prompt(refined_query, schema_info or {})

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response
        return self._parse_cypher_response(response)

    def _parse_cypher_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for Cypher generation.

        Args:
            response: LLM response text

        Returns:
            Dictionary with Cypher query and metadata
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
                "cypher_query": data.get("cypher_query", ""),
                "query_explanation": data.get("query_explanation", ""),
                "expected_result_type": data.get("expected_result_type", "unknown"),
                "result_structure": data.get("result_structure", ""),
            }

        except json.JSONDecodeError:
            # Fallback: return empty query
            print("Warning: Failed to parse Cypher generation response")
            return {
                "cypher_query": "",
                "query_explanation": "Failed to parse response",
                "expected_result_type": "unknown",
                "result_structure": "",
            }
        except Exception as e:
            print(f"Warning: Failed to parse Cypher generation response: {e}")
            return {
                "cypher_query": "",
                "query_explanation": f"Error: {e}",
                "expected_result_type": "unknown",
                "result_structure": "",
            }

    def generate_cypher_with_entities(
        self,
        refined_query: str,
        entity_matches: List[Dict[str, Any]],
        predicate_matches: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher query using matched entities and predicates.

        Args:
            refined_query: The refined user query
            entity_matches: List of matched entities from Qdrant
            predicate_matches: List of matched predicates from Qdrant

        Returns:
            Dictionary with Cypher query and metadata
        """
        # Format entity information
        entity_info = []
        for match in entity_matches[:5]:  # Use top 5 matches
            entity_info.append({
                "name": match["payload"].get("name", ""),
                "type": match["payload"].get("type", ""),
                "score": match["score"]
            })

        # Format predicate information
        predicate_info = []
        if predicate_matches:
            for match in predicate_matches[:3]:  # Use top 3 matches
                predicate_info.append({
                    "name": match["payload"].get("name", ""),
                    "score": match["score"]
                })

        # Create prompt with entity context
        prompt = f"""You are an expert in Neo4j Cypher query generation and knowledge graph querying.

Your task is to generate a valid Cypher query to answer the user's question using the matched entities and predicates.

**User Query:**
{refined_query}

**Matched Entities:**
{json.dumps(entity_info, indent=2)}

**Matched Predicates:**
{json.dumps(predicate_info, indent=2)}

**Graph Schema Information:**
- Node labels: Entity
- Node properties: name, type, canonical_id, updated_at
- Relationship types: Various predicates (dynamic based on extracted triples)
- Relationship properties: chunk_id, updated_at, and other extracted properties

**IMPORTANT Cypher Syntax Rules:**
1. Use proper MATCH patterns: `MATCH (n:Entity)-[r]-(m) WHERE n.name = 'value' RETURN n, r, m`
2. For finding entities by name: `MATCH (n:Entity) WHERE n.name CONTAINS 'value' RETURN n`
3. For getting relationships: `MATCH (n:Entity)-[r]-(m) WHERE n.name = 'value' RETURN n, r, m`
4. NEVER use `relationships()` function in RETURN clause - it's invalid syntax
5. Use proper relationship patterns: `(n)-[r:PREDICATE]->(m)` for directed relationships
6. Use `(n)-[r]-(m)` for undirected relationships
7. Always include proper RETURN clause with node/relationship variables

**Instructions:**
1. Use the matched entities to construct specific WHERE clauses
2. Use the matched predicates to construct specific relationship patterns
3. Generate a query that retrieves relevant information about the matched entities
4. Include relationships and connected nodes to provide context
5. Use CONTAINS for flexible name matching
6. Limit results to avoid overwhelming responses (LIMIT 50)

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH (n:Entity)-[r]-(m) WHERE n.name CONTAINS 'example' RETURN n, r, m LIMIT 50",
  "query_explanation": "brief explanation of what the query does",
  "expected_result_type": "nodes | relationships | paths | aggregates",
  "result_structure": "description of the expected result structure"
}}
```

Generate the Cypher query and return the result in JSON format:"""

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response
        return self._parse_cypher_response(response)

    def generate_cypher_from_nodes(
        self,
        user_query: str,
        high_connectivity_nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate Cypher query based on high connectivity nodes.

        Args:
            user_query: The original user query
            high_connectivity_nodes: List of high connectivity nodes

        Returns:
            Dictionary with Cypher query and metadata
        """
        # Format node information
        node_info = []
        for node in high_connectivity_nodes:
            node_info.append({
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "connections": node.get("connection_count", 0)
            })

        # Create prompt with node context
        prompt = f"""You are an expert in Neo4j Cypher query generation and knowledge graph querying.

Your task is to generate a Cypher query to explore the knowledge graph using high-connectivity nodes, based on the user's query intent.

**User Query:**
{user_query}

**Available High-Connectivity Nodes:**
{json.dumps(node_info, indent=2)}

**Graph Schema Information:**
- Node labels: Entity
- Node properties: name, type, canonical_id, updated_at
- Relationship types: Various predicates (dynamic based on extracted triples)
- Relationship properties: chunk_id, updated_at, and other extracted properties

**IMPORTANT Cypher Syntax Rules:**
1. Use proper MATCH patterns: `MATCH (n:Entity)-[r]-(m) WHERE n.name = 'value' RETURN n, r, m`
2. For finding entities by name: `MATCH (n:Entity) WHERE n.name CONTAINS 'value' RETURN n`
3. For getting relationships: `MATCH (n:Entity)-[r]-(m) WHERE n.name = 'value' RETURN n, r, m`
4. NEVER use `relationships()` function in RETURN clause - it's invalid syntax
5. Use proper relationship patterns: `(n)-[r:PREDICATE]->(m)` for directed relationships
6. Use `(n)-[r]-(m)` for undirected relationships
7. Always include proper RETURN clause with node/relationship variables

**Instructions:**
1. Analyze the user's query to understand their intent
2. Select 2-3 most relevant high-connectivity nodes based on the query
3. Generate a query that explores relationships around these nodes
4. Include connected nodes and relationships to provide comprehensive context
5. Use CONTAINS for flexible name matching
6. Limit results to avoid overwhelming responses (LIMIT 50)

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH (n:Entity)-[r]-(m) WHERE n.name CONTAINS 'example' RETURN n, r, m LIMIT 50",
  "query_explanation": "brief explanation of what the query does",
  "expected_result_type": "nodes | relationships | paths | aggregates",
  "result_structure": "description of the expected result structure"
}}
```

Generate the Cypher query and return the result in JSON format:"""

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response
        return self._parse_cypher_response(response)

    def generate_high_connectivity_query(self) -> Dict[str, Any]:
        """Generate Cypher query to find high connectivity nodes.

        Returns:
            Dictionary with Cypher query and metadata
        """
        # Create prompt
        prompt = get_high_connectivity_query_prompt()

        # Get LLM response
        response = self._get_llm_response(prompt)

        # Parse response
        return self._parse_cypher_response(response)
