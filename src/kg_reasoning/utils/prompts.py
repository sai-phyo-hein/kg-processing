"""Prompt management module for knowledge graph reasoning."""

from typing import Dict, List


def get_entity_extraction_prompt(
    user_query: str,
    entity_matches: List[Dict] = None,
    predicate_matches: List[Dict] = None,
    keywords: List[str] = None,
) -> str:
    """Get prompt for extracting entities from user query with hybrid Qdrant matches.

    Args:
        user_query: The user's natural language query
        entity_matches: List of matched entities from Qdrant
        predicate_matches: List of matched predicates from Qdrant
        keywords: List of extracted keywords from the query

    Returns:
        Prompt string for entity extraction
    """
    # Format keywords
    keywords_str = ", ".join(keywords) if keywords else "No keywords extracted"

    # Format entity matches with match type information
    entity_matches_str = ""
    if entity_matches:
        entity_matches_str = "\n".join([
            f"- {match['payload'].get('name', 'Unknown')} (type: {match['payload'].get('type', 'Unknown')}, "
            f"combined_score: {match['score']:.2f}, keyword_score: {match.get('keyword_score', 0):.2f}, "
            f"semantic_score: {match.get('semantic_score', 0):.2f}, match_type: {', '.join(match.get('match_type', []))})"
            for match in entity_matches
        ])
    else:
        entity_matches_str = "No entity matches found"

    # Format predicate matches with match type information
    predicate_matches_str = ""
    if predicate_matches:
        predicate_matches_str = "\n".join([
            f"- {match['payload'].get('name', 'Unknown')} (combined_score: {match['score']:.2f}, "
            f"keyword_score: {match.get('keyword_score', 0):.2f}, semantic_score: {match.get('semantic_score', 0):.2f}, "
            f"match_type: {', '.join(match.get('match_type', []))})"
            for match in predicate_matches
        ])
    else:
        predicate_matches_str = "No predicate matches found"

    return f"""You are an expert in entity extraction and knowledge graph querying.

Your task is to extract key entities and predicates from the user's query that would be relevant for querying a knowledge graph, using hybrid matching results as context.

**User Query:**
{user_query}

**Extracted Keywords:**
{keywords_str}

**Hybrid Qdrant Entity Matches:**
{entity_matches_str}

**Hybrid Qdrant Predicate Matches:**
{predicate_matches_str}

**Matching Strategy:**
- Keyword matches have higher precision (exact or near-exact terminology)
- Semantic matches provide broader context and handle synonyms
- Combined scores reflect both precision and semantic relevance
- Prioritize matches with higher combined scores and keyword matches

**Instructions:**
1. Analyze the user query to identify the main entities and predicates being asked about
2. Use the hybrid Qdrant matches as context to identify the most relevant entities and predicates
3. Prioritize keyword matches for precise terminology, but consider semantic matches for broader context
4. Select entities and predicates that have high combined scores and are relevant to the query
5. If no relevant matches are found, extract entities and predicates directly from the query
6. Focus on entities and predicates that would be useful for constructing a knowledge graph query
7. Consider the match_type information - keyword matches are more precise, semantic matches are more flexible

**Output Format (JSON):**
```json
{{
  "entities": [
    {{
      "name": "entity_name_from_qdrant_or_query",
      "type": "entity_type",
      "context": "brief context about this entity in the query",
      "source": "keyword | semantic | extracted",
      "combined_score": 0.95,
      "match_type": ["keyword", "semantic"]
    }}
  ],
  "predicates": [
    {{
      "name": "predicate_name_from_qdrant_or_query",
      "context": "brief context about this predicate in the query",
      "source": "keyword | semantic | extracted",
      "combined_score": 0.88,
      "match_type": ["keyword"]
    }}
  ],
  "query_intent": "brief description of what the user is asking for",
  "query_type": "factual | causal | comparative | exploratory"
}}
```

Extract the entities and predicates, prioritizing keyword matches for precision while using semantic matches for context, and return the result in JSON format:"""


def get_query_refinement_prompt(
    original_query: str,
    entity_matches: List[Dict],
) -> str:
    """Get prompt for refining user query with canonical entities.

    Args:
        original_query: The original user query
        entity_matches: List of matched canonical entities from Qdrant

    Returns:
        Prompt string for query refinement
    """
    matches_str = "\n".join([
        f"- {match['original']} → {match['canonical']} (similarity: {match['score']:.2f})"
        for match in entity_matches
    ])

    return f"""You are an expert in query refinement and knowledge graph querying.

Your task is to refine the user's query by replacing entity mentions with their canonical forms found in the knowledge graph.

**Original Query:**
{original_query}

**Entity Matches Found:**
{matches_str}

**Instructions:**
1. Replace entity mentions in the original query with their canonical forms
2. Maintain the original intent and meaning of the query
3. Ensure the refined query is natural and grammatically correct
4. If an entity has multiple possible matches, choose the one with highest similarity
5. Keep the query structure and flow natural

**Output Format (JSON):**
```json
{{
  "refined_query": "refined query with canonical entities",
  "entity_replacements": [
    {{
      "original": "original_entity",
      "canonical": "canonical_entity",
      "similarity": 0.95
    }}
  ],
  "reasoning": "brief explanation of the refinement"
}}
```

Refine the query and return the result in JSON format:"""


def get_cypher_generation_prompt(
    refined_query: str,
    schema_info: Dict,
) -> str:
    """Get prompt for generating Cypher query.

    Args:
        refined_query: The refined user query with canonical entities
        schema_info: Information about the graph schema

    Returns:
        Prompt string for Cypher generation
    """
    return f"""You are an expert in Neo4j Cypher query generation and knowledge graph querying.

Your task is to generate a valid Cypher query to answer the user's question based on the refined query.

**Refined Query:**
{refined_query}

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
1. Analyze the refined query to understand what information is being requested
2. Generate a VALID Cypher query that retrieves the relevant information
3. Use appropriate Cypher patterns (MATCH, WHERE, RETURN, etc.)
4. Handle entity matching using the canonical_id or name properties
5. Include relevant relationships and properties in the query
6. Optimize the query for performance (use indexes where appropriate)
7. Return results in a structured format that can be easily interpreted
8. TEST your query mentally - ensure it follows proper Cypher syntax

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH (n:Entity)-[r]-(m) WHERE n.name CONTAINS 'example' RETURN n, r, m",
  "query_explanation": "brief explanation of what the query does",
  "expected_result_type": "nodes | relationships | paths | aggregates",
  "result_structure": "description of the expected result structure"
}}
```

Generate the Cypher query and return the result in JSON format:"""


def get_answer_synthesis_prompt(
    user_query: str,
    neo4j_results: List[Dict],
    cypher_query: str,
) -> str:
    """Get prompt for synthesizing answer from Neo4j results.

    Args:
        user_query: The original user query
        neo4j_results: Results from Neo4j query
        cypher_query: The Cypher query that was executed

    Returns:
        Prompt string for answer synthesis
    """
    results_str = "\n".join([
        f"- Result {i+1}: {result}"
        for i, result in enumerate(neo4j_results[:10])  # Limit to first 10 results
    ])

    return f"""You are an expert in knowledge graph reasoning and answer synthesis.

Your task is to synthesize a clear, accurate answer to the user's question based ONLY on the Neo4j query results.

**User Query:**
{user_query}

**Cypher Query Executed:**
{cypher_query}

**Neo4j Results:**
{results_str}

**CRITICAL INSTRUCTIONS:**
1. Use ONLY the information provided in the Neo4j results above
2. Do NOT use external knowledge, general information, or make assumptions
3. If the results are empty or don't contain relevant information, state this clearly
4. Do NOT provide general knowledge or common information as a fallback
5. Base your answer ENTIRELY on the actual query results shown above

**Answer Structure:**
1. **Direct Answer**: Answer the user's question using ONLY the results
2. **Supporting Evidence**: Quote specific data from the results
3. **Context**: Explain what the results show (based on actual data)
4. **Limitations**: If results are incomplete, acknowledge this

**If Results Are Empty:**
- State clearly: "The query did not return any results from the knowledge graph."
- Do NOT provide general knowledge or common information
- Suggest trying a different query or entity

**Output Format:**
Provide a natural language answer that directly addresses the user's question using ONLY the provided results. Include:
- A direct answer based on the results
- Specific data points from the results
- Clear attribution to the query results
- Acknowledgment of any limitations in the results

Synthesize the answer:"""


def get_query_suggestion_prompt(
    user_query: str,
    high_connectivity_nodes: List[Dict],
) -> str:
    """Get prompt for suggesting alternative queries when no entity matches found.

    Args:
        user_query: The original user query
        high_connectivity_nodes: List of high connectivity nodes from Neo4j

    Returns:
        Prompt string for query suggestions
    """
    nodes_str = "\n".join([
        f"- {node['name']} (type: {node['type']}, connections: {node['connection_count']})"
        for node in high_connectivity_nodes[:15]  # Limit to top 15
    ])

    return f"""You are an expert in knowledge graph exploration and query assistance.

The user's query did not match any entities in the knowledge graph. Your task is to suggest alternative queries based on the available high-connectivity entities.

**User Query:**
{user_query}

**Available High-Connectivity Entities:**
{nodes_str}

**Instructions:**
1. Analyze the user's query to understand their intent
2. Identify relevant entities from the available high-connectivity nodes
3. Suggest 3-5 alternative queries that the user might be interested in
4. Each suggestion should be a natural language question
5. Explain why each suggestion might be relevant
6. Group suggestions by topic or theme if possible

**Output Format (JSON):**
```json
{{
  "message": "No direct matches found for your query. Here are some alternative queries you might be interested in:",
  "suggestions": [
    {{
      "query": "alternative natural language query",
      "reason": "why this suggestion is relevant",
      "entities": ["entity1", "entity2"]
    }}
  ],
  "topics": ["topic1", "topic2"]
}}
```

Generate query suggestions and return the result in JSON format:"""


def get_high_connectivity_query_prompt() -> str:
    """Get prompt for generating Cypher query to find high connectivity nodes.

    Returns:
        Prompt string for high connectivity query generation
    """
    return """You are an expert in Neo4j Cypher query generation.

Your task is to generate a Cypher query to find high-connectivity nodes in the knowledge graph.

**Instructions:**
1. Find nodes with the highest number of relationships (both incoming and outgoing)
2. Return the top nodes with their connection counts
3. Include node properties like name, type, and canonical_id
4. Order by connection count in descending order
5. Limit results to a reasonable number (e.g., 20)

**Output Format (JSON):**
```json
{{
  "cypher_query": "MATCH (n:Entity)-[r]-() RETURN n.name, n.type, count(r) as connections ORDER BY connections DESC LIMIT 20",
  "query_explanation": "brief explanation of what the query does"
}}
```

Generate the Cypher query and return the result in JSON format:"""
