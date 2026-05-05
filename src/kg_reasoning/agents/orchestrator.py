"""Orchestrator agent for multi-agent reasoning system.

Analyzes user queries, retrieves canonical IDs from Qdrant, and plans
up to 5 different query strategies for worker agents to execute.
"""

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from kg_reasoning.agents.tools.qdrant_tools import (
    get_canonical_entities,
    get_canonical_predicates,
    get_community_metadata,
)
from kg_reasoning.agents.tools.neo4j_tools import get_top_connected_nodes


ORCHESTRATOR_SYSTEM_PROMPT = """You are an Orchestrator Agent for a knowledge graph reasoning system.

Your role is to:
1. Analyze the user's question to identify key entities, relationships, and concepts
2. Use Qdrant tools to find canonical IDs for entities and predicates
3. Check community metadata to understand how the graph is organized
4. Plan up to 5 different query strategies that could answer the question

## Available Tools

You have access to these tools:
- get_canonical_entities: Find canonical entity IDs from entity_registry
- get_canonical_predicates: Find canonical predicate IDs from predicate_registry
- get_community_metadata: Get information about graph communities
- get_top_connected_nodes: Get the 20 most connected nodes in the graph

## Qdrant Payload Schema

Both entity_registry and predicate_registry points have the following payload fields:
- **canonical_id** (uuid): Use this UUID as the stable identifier when building strategies
- **name** (keyword): Human-readable canonical name; use this for display and keyword filtering

When interpreting tool results, always extract `canonical_id` (not any other ID field) for use
in strategy `canonical_ids`, and use `name` when you need a human-readable label.

## Query Strategy Planning

For each strategy, specify:
- **name**: Short descriptive name (e.g., "direct_entity_query", "expanded_relationships")
- **description**: What this strategy aims to find
- **approach**: High-level approach (e.g., "direct match", "graph expansion", "path finding")
- **canonical_ids**: Relevant entity/predicate canonical IDs to use
- **community_ids**: Relevant community IDs to filter by (if applicable)
- **parameters**: Additional parameters like depth, relationship types, etc.

## Fallback When No Entities or Predicates Are Found

If get_canonical_entities and get_canonical_predicates return no matches:
1. Call get_top_connected_nodes to retrieve the 20 most connected nodes
2. Use those nodes as entry points — treat their canonical_ids as the seed entities
3. Build strategies that explore the graph from these high-connectivity nodes
   (e.g., expand their relationships, find communities they belong to, trace paths between them)
4. Do NOT emit a bare "fallback_query" with empty canonical_ids

## Important Notes

- Edges in the graph can be differentiated by community_id in their metadata
- Use community_metadata_registry to understand which communities might be relevant
- Plan diverse strategies: some direct, some exploratory, some focused on relationships
- Maximum 5 strategies - prioritize the most promising approaches
- Each strategy should be distinct and offer a different perspective

Output your final plan as a structured JSON object with a "strategies" list."""


class OrchestratorAgent:
    """Orchestrator agent for query analysis and strategy planning."""

    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        temperature: float = 0.1,
    ):
        """Initialize the orchestrator agent.

        Args:
            llm_provider: LLM provider (currently only "openai" supported)
            llm_model: Model name
            temperature: LLM temperature for response generation
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Setup tools
        self.tools = [
            get_canonical_entities,
            get_canonical_predicates,
            get_community_metadata,
            get_top_connected_nodes,
        ]

        # Create agent using langgraph
        self.agent = create_react_agent(self.llm, self.tools)

    def analyze_and_plan(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and plan query strategies.

        Args:
            user_query: The user's natural language question

        Returns:
            Dictionary containing strategies and metadata
        """
        # Add explicit instructions for output format
        enhanced_query = f"""{ORCHESTRATOR_SYSTEM_PROMPT}

User Question: {user_query}

Please:
1. Use the tools to find canonical entities and predicates for this question
2. Check community metadata if relevant
3. Plan up to 5 query strategies

Output your final plan in this JSON format:
{{
  "entities_found": [...list of canonical entities...],
  "predicates_found": [...list of canonical predicates...],
  "communities_identified": [...list of relevant community IDs...],
  "strategies": [
    {{
      "name": "strategy_name",
      "description": "what this strategy finds",
      "approach": "direct/expansion/path",
      "canonical_ids": {{
        "entities": [...],
        "predicates": [...]
      }},
      "community_ids": [...],
      "parameters": {{
        "depth": 1,
        "relationship_types": [...]
      }}
    }}
  ]
}}"""

        # Execute agent
        messages = [HumanMessage(content=enhanced_query)]
        result = self.agent.invoke({"messages": messages})

        # Extract output from messages
        output = ""
        intermediate_steps = []
        
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "content"):
                    output += msg.content + "\n"
                # Track tool calls as intermediate steps
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        intermediate_steps.append((tool_call, None))

        # Try to parse JSON from output
        import json
        import re

        # Find JSON in output
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Fallback: create an exploratory plan seeded by top connected nodes
                plan = self._create_connectivity_fallback_plan()
        else:
            # No JSON found, create basic plan from tool outputs
            plan = self._create_basic_plan_from_steps(intermediate_steps)

        # Ensure we have strategies
        if "strategies" not in plan or not plan["strategies"]:
            fallback = self._create_connectivity_fallback_plan()
            plan["strategies"] = fallback["strategies"]

        # Limit to 5 strategies
        plan["strategies"] = plan["strategies"][:5]

        # Add metadata
        plan["raw_output"] = output
        plan["user_query"] = user_query

        return plan

    def _create_connectivity_fallback_plan(self) -> Dict[str, Any]:
        """Build a fallback plan seeded by the top 20 most connected nodes.

        Called when the LLM output cannot be parsed or contains no strategies.

        Returns:
            Plan dictionary with connectivity-based strategies.
        """
        import json
        from kg_reasoning.agents.tools.neo4j_tools import _get_manager as _get_neo4j_manager

        try:
            manager = _get_neo4j_manager()
            cypher = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC
            LIMIT 20
            RETURN n.canonical_id AS canonical_id, n.name AS name, degree
            """
            rows = manager.execute_cypher_query(cypher, {}, 20)
            top_ids = [r["canonical_id"] for r in rows if r.get("canonical_id")]
        except Exception:
            top_ids = []

        return {
            "entities_found": top_ids,
            "predicates_found": [],
            "communities_identified": [],
            "strategies": [
                {
                    "name": "top_connected_expansion",
                    "description": "Explore relationships of the most connected nodes as entry points",
                    "approach": "expansion",
                    "canonical_ids": {"entities": top_ids, "predicates": []},
                    "community_ids": [],
                    "parameters": {"depth": 1, "limit": 50},
                },
                {
                    "name": "top_connected_direct",
                    "description": "Retrieve direct triples involving the most connected nodes",
                    "approach": "direct",
                    "canonical_ids": {"entities": top_ids, "predicates": []},
                    "community_ids": [],
                    "parameters": {"limit": 50},
                },
            ],
        }

    def _create_basic_plan_from_steps(
        self, intermediate_steps: List[tuple]
    ) -> Dict[str, Any]:
        """Create a basic plan from intermediate tool outputs.

        Args:
            intermediate_steps: List of (action, observation) tuples

        Returns:
            Basic plan dictionary
        """
        import json

        entities = []
        predicates = []
        communities = []

        # Extract information from tool outputs
        for action, observation in intermediate_steps:
            tool_name = action.tool if hasattr(action, 'tool') else ""

            try:
                obs_data = json.loads(observation) if isinstance(observation, str) else observation

                if "canonical_entities" in tool_name or isinstance(obs_data, list):
                    for item in obs_data if isinstance(obs_data, list) else []:
                        if "canonical_id" in item:
                            entities.append(item["canonical_id"])

                if "canonical_predicates" in tool_name:
                    for item in obs_data if isinstance(obs_data, list) else []:
                        if "canonical_id" in item:
                            predicates.append(item["canonical_id"])

                if "community_metadata" in tool_name:
                    for item in obs_data if isinstance(obs_data, list) else []:
                        if "community_id" in item:
                            communities.append(item["community_id"])
            except Exception:
                pass

        return {
            "entities_found": list(set(entities)),
            "predicates_found": list(set(predicates)),
            "communities_identified": list(set(communities)),
            "strategies": [{
                "name": "entity_based_query",
                "description": "Query based on found entities and predicates",
                "approach": "direct",
                "canonical_ids": {
                    "entities": list(set(entities)),
                    "predicates": list(set(predicates)),
                },
                "community_ids": list(set(communities)),
                "parameters": {"limit": 50},
            }]
        }
