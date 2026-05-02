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
- Relationship properties:
  * community_id: Unique identifier for the community/village where this relationship was extracted
  * chunk_id: Chunk identifier from the source document
  * updated_at: Timestamp of when the relationship was created/updated
  * other extracted properties: Various domain-specific properties from the extraction

**Filtering by Community:**
You can filter relationships by community using: `WHERE r.community_id = 'community_identifier'`
Example: `MATCH (n:Entity)-[r]-(m) WHERE r.community_id = '1_บ้านหนองกอก' RETURN n, r, m`

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
- Relationship properties:
  * community_id: Unique identifier for the community/village where this relationship was extracted
  * chunk_id: Chunk identifier from the source document
  * updated_at: Timestamp of when the relationship was created/updated
  * other extracted properties: Various domain-specific properties from the extraction

**Filtering by Community:**
You can filter relationships by community using: `WHERE r.community_id = 'community_identifier'`
Example: `MATCH (n:Entity)-[r]-(m) WHERE r.community_id = '1_บ้านหนองกอก' RETURN n, r, m`

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

    def generate_multiple_cypher_queries(
        self,
        refined_query: str,
        entity_matches: List[Dict[str, Any]],
        predicate_matches: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple Cypher queries: one merged and individual queries for each entity/predicate.

        Args:
            refined_query: The refined user query
            entity_matches: List of matched entities from Qdrant
            predicate_matches: List of matched predicates from Qdrant

        Returns:
            List of dictionaries with Cypher queries and metadata
        """
        queries = []

        # 1. Generate merged query with all entities and predicates
        merged_query = self.generate_cypher_with_entities(
            refined_query, entity_matches, predicate_matches
        )
        merged_query["query_type"] = "merged"
        merged_query["description"] = "Merged query with all entities and predicates"
        queries.append(merged_query)

        # 2. Generate individual queries for each entity
        for idx, entity_match in enumerate(entity_matches[:5]):  # Limit to top 5
            entity_name = entity_match["payload"].get("name", "")
            entity_type = entity_match["payload"].get("type", "")

            if not entity_name:
                continue

            # Generate query for this specific entity
            single_entity_query = self._generate_single_entity_query(
                refined_query, entity_name, entity_type
            )
            single_entity_query["query_type"] = "individual_entity"
            single_entity_query["description"] = f"Query for entity: {entity_name}"
            single_entity_query["entity_name"] = entity_name
            queries.append(single_entity_query)

        # 3. Generate individual queries for each predicate
        if predicate_matches:
            for idx, predicate_match in enumerate(predicate_matches[:3]):  # Limit to top 3
                predicate_name = predicate_match["payload"].get("name", "")

                if not predicate_name:
                    continue

                # Generate query for this specific predicate
                single_predicate_query = self._generate_single_predicate_query(
                    refined_query, predicate_name
                )
                single_predicate_query["query_type"] = "individual_predicate"
                single_predicate_query["description"] = f"Query for predicate: {predicate_name}"
                single_predicate_query["predicate_name"] = predicate_name
                queries.append(single_predicate_query)

        return queries

    def _generate_single_entity_query(
        self,
        refined_query: str,
        entity_name: str,
        entity_type: str,
    ) -> Dict[str, Any]:
        """Generate a Cypher query for a single entity.

        Args:
            refined_query: The refined user query
            entity_name: Name of the entity
            entity_type: Type of the entity

        Returns:
            Dictionary with Cypher query and metadata
        """
        prompt = f"""You are an expert in Neo4j Cypher query generation.

Generate a Cypher query to retrieve information about a specific entity and its relationships.

**User Query:**
{refined_query}

**Target Entity:**
- Name: {entity_name}
- Type: {entity_type}

**Graph Schema:**
- Node labels: Entity
- Node properties: name, type, canonical_id, updated_at
- Relationship types: Various predicates (dynamic)

**IMPORTANT Cypher Syntax Rules:**
1. Use proper MATCH patterns: `MATCH (n:Entity)-[r]-(m) WHERE n.name = 'value' RETURN n, r, m`
2. For finding entities by name: `MATCH (n:Entity) WHERE n.name CONTAINS 'value' RETURN n`
3. NEVER use `relationships()` function
4. Use `(n)-[r]-(m)` for undirected relationships

**Instructions:**
1. Generate a query that finds the entity by name
2. Include its relationships and connected nodes
3. Use CONTAINS for flexible name matching
4. Limit results (LIMIT 20)

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH (n:Entity)-[r]-(m) WHERE n.name CONTAINS '{entity_name}' RETURN n, r, m LIMIT 20",
  "query_explanation": "Retrieves entity '{entity_name}' and its relationships",
  "expected_result_type": "relationships",
  "result_structure": "entity node, relationships, connected nodes"
}}
```

Generate the Cypher query:"""

        response = self._get_llm_response(prompt)
        return self._parse_cypher_response(response)

    def _generate_single_predicate_query(
        self,
        refined_query: str,
        predicate_name: str,
    ) -> Dict[str, Any]:
        """Generate a Cypher query for a specific predicate/relationship.

        Args:
            refined_query: The refined user query
            predicate_name: Name of the predicate/relationship

        Returns:
            Dictionary with Cypher query and metadata
        """
        prompt = f"""You are an expert in Neo4j Cypher query generation.

Generate a Cypher query to find all relationships of a specific type.

**User Query:**
{refined_query}

**Target Relationship Type:**
{predicate_name}

**Graph Schema:**
- Node labels: Entity
- Relationship types: Various predicates (dynamic)

**IMPORTANT Cypher Syntax Rules:**
1. Use proper MATCH patterns with relationship type: `MATCH (n:Entity)-[r:{predicate_name}]-(m) RETURN n, r, m`
2. NEVER use `relationships()` function
3. Include both source and target nodes

**Instructions:**
1. Generate a query that finds all relationships of type '{predicate_name}'
2. Include source and target nodes
3. Limit results (LIMIT 20)

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH (n:Entity)-[r:`{predicate_name}`]-(m) RETURN n, r, m LIMIT 20",
  "query_explanation": "Retrieves all relationships of type '{predicate_name}'",
  "expected_result_type": "relationships",
  "result_structure": "source node, relationship, target node"
}}
```

Generate the Cypher query:"""

        response = self._get_llm_response(prompt)
        return self._parse_cypher_response(response)
