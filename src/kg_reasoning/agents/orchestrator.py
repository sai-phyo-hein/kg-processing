"""Orchestrator agent for multi-agent reasoning system.

Analyzes user queries, retrieves canonical IDs from Qdrant, and plans
up to 5 different query strategies for worker agents to execute.
"""

from typing import Any, Dict, List, Optional

from kg_extractor.utils.model_setup import REASONING_PROVIDER, ORCHESTRATOR_MODEL, get_reasoning_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from kg_reasoning.agents.tools.qdrant_tools import (
    get_canonical_entities,
    get_canonical_predicates,
    get_community_metadata,
    get_labels_by_group,
    get_entities_by_labels,
    get_all_community_ids,
)
from kg_reasoning.agents.tools.neo4j_tools import (
    get_top_connected_nodes,
    get_community_ids_from_relationships,
)


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
- get_labels_by_group: Get labels filtered by label group (e.g., "Community", "Person")
- get_entities_by_labels: Get entities filtered by specific label names
- get_all_community_ids: Get list of all available community IDs in the graph
- get_top_connected_nodes: Get the 20 most connected nodes in the graph
- get_community_ids_from_relationships: Get actual community IDs from Neo4j relationship properties

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
- **approach**: High-level approach (e.g., "direct", "expansion", "path")
- **canonical_ids**: Relevant entity/predicate canonical IDs to use
- **community_ids**: Relevant community IDs to filter by (if applicable)
- **parameters**: Additional parameters like depth, relationship types, etc.

## Strategy Types to Include

When creating strategies, ensure you cover multiple query approaches:
1. **Direct Connections**: Direct triples/relationships involving the entities (approach: "direct")
2. **Graph Expansion**: 1-hop or 2-hop neighborhood exploration (approach: "expansion")
3. **Predicate-based**: Connections using specific relationship types (approach: "direct" with predicates)
4. **Path Finding**: Paths between entities if 2+ entities found (approach: "path")
5. **Community Exploration**: For broad community queries, query all relationships filtered by community_id (approach: "community_exploration")

This multi-faceted approach ensures comprehensive coverage of the knowledge graph.

## Community-Based Query Filtering

**IMPORTANT**: If the user's query is community-based (mentions specific village, "moo"/"หมู่", sub-district, or location):

1. **Get available community IDs**: FIRST call `get_community_ids_from_relationships()` to see what community IDs exist in Neo4j relationships
2. **Match the query to a community_id**: Match the user's query to available community IDs (e.g., "หมู่ 1" might match "หมู่ 1_village" or similar)

### Broad vs Specific Community Queries

**Broad Community Query** (e.g., "What can we know about หมู่ 1?", "Tell me about Moo 1"):
- The user wants comprehensive information about a community
- Create 3 PARALLEL strategy approaches:
  1. **Entity-based**: Query entities that match the community name
  2. **Label-based**: Query entities by relevant label groups (Community, Person, Activity, Organization)
  3. **Community-filtered**: Query ALL relationships where `r.community_id` matches the community
- All 3 strategies should include the matched `community_id` in the `community_ids` list

**Specific Community Query** (e.g., "Who are the leaders in หมู่ 1?", "What activities happen in Moo 1?"):
- The user has a specific question about the community
- Use regular entity/predicate-based strategies with community_id filter

**Neo4j Relationship Structure**:
- Relationships (predicates) have a `community_id` property
- Example: `(subject)-[r]->(object)` where `r.community_id = "หมู่ 1_village"`
- Worker agents will filter relationships using: `WHERE r.community_id IN $community_ids`

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
        llm_provider: str = REASONING_PROVIDER,
        llm_model: str = ORCHESTRATOR_MODEL,
        temperature: float = 0.1,
    ):
        """Initialize the orchestrator agent.

        Args:
            llm_provider: LLM provider (supports: openai, openrouter, groq, nvidia)
            llm_model: Model name
            temperature: LLM temperature for response generation
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize LLM using helper function that supports multiple providers
        self.llm = get_reasoning_llm(model=llm_model, temperature=temperature)

        # Setup tools
        self.tools = [
            get_canonical_entities,
            get_canonical_predicates,
            get_community_metadata,
            get_labels_by_group,
            get_entities_by_labels,
            get_all_community_ids,
            get_top_connected_nodes,
            get_community_ids_from_relationships,
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
1. **Detect if this is a broad community query**: Check if the query is asking for general/comprehensive information 
   about a community (e.g., "What can we know about หมู่ 1?", "Tell me about Moo 1") vs a specific question.
   
2. **Get community IDs if community-based**: If the question mentions a specific village or moo/หมู่, 
   call `get_community_ids_from_relationships()` to see available communities, then match the query 
   to the correct community_id.
   
3. **For BROAD community queries, create 3 parallel strategies**:
   a. Entity-based: Find entities matching the community name and query their connections
   b. Label-based: Get entities from relevant label groups (Community, Person, Activity, etc.)
   c. Community-exploration: Query ALL relationships filtered by the community_id (approach: "community_exploration")
   
4. **For specific community queries**: Use regular entity/predicate-based strategies with community_id filter

5. **For non-community queries**: Use the tools to find canonical entities and predicates, then plan diverse strategies

**IMPORTANT**: Return ONLY valid JSON with actual data from tool results, not template placeholders.

Output your final plan in this JSON format:
{{
  "query_type": "broad_community" | "specific_community" | "general",
  "entities_found": ["uuid-1", "uuid-2"],
  "predicates_found": ["uuid-3"],
  "communities_identified": ["community_id_1"],
  "strategies": [
    {{
      "name": "entity_direct_query",
      "description": "Query direct connections of Community entities",
      "approach": "direct",
      "canonical_ids": {{
        "entities": ["uuid-1", "uuid-2"],
        "predicates": []
      }},
      "community_ids": ["community_id_1"],
      "parameters": {{"limit": 50}}
    }}
  ]
}}

Replace the example UUIDs above with actual canonical_ids from your tool calls. Do NOT use placeholder text like "[...list...]" - use real values."""

        # Execute agent
        messages = [HumanMessage(content=enhanced_query)]
        result = self.agent.invoke({"messages": messages})

        # Extract output from messages, capturing tool calls AND tool results
        import json
        import re
        from langchain_core.messages import AIMessage, ToolMessage

        output = ""
        intermediate_steps = []
        pending_tool_calls: dict = {}  # tool_call_id -> tool_call

        print("\n[Orchestrator Debug] ── Message trace ──────────────────────────")
        if "messages" in result:
            for msg in result["messages"]:
                msg_type = type(msg).__name__

                # Accumulate final text output
                if hasattr(msg, "content") and msg.content:
                    output += msg.content + "\n"

                # AIMessage: may contain tool calls
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            pending_tool_calls[tc["id"]] = tc
                            print(f"  [tool_call] {tc['name']}  args={tc['args']}")
                    elif msg.content:
                        # Final text response from the LLM
                        preview = msg.content[:300].replace("\n", " ")
                        print(f"  [ai_response] {preview}{'...' if len(msg.content) > 300 else ''}")

                # ToolMessage: result returned by a tool
                elif isinstance(msg, ToolMessage):
                    tc = pending_tool_calls.pop(msg.tool_call_id, None)
                    tool_name = tc["name"] if tc else msg.name or "unknown_tool"
                    try:
                        tool_result = json.loads(msg.content)
                        result_count = len(tool_result) if isinstance(tool_result, list) else "non-list"
                        print(f"  [tool_result] {tool_name} → {result_count} results")
                        if isinstance(tool_result, list) and tool_result:
                            for item in tool_result[:3]:
                                # Handle both dict results (entities/predicates) and string results (community IDs)
                                if isinstance(item, dict):
                                    print(f"    • canonical_id={item.get('canonical_id')} name={item.get('name')} score={item.get('score')}")
                                else:
                                    # Plain string or other type
                                    print(f"    • {item}")
                            if len(tool_result) > 3:
                                print(f"    … and {len(tool_result) - 3} more")
                                print(f"    … and {len(tool_result) - 3} more")
                        elif isinstance(tool_result, list):
                            print(f"    (empty list — no matches above threshold)")
                    except (json.JSONDecodeError, TypeError):
                        preview = str(msg.content)[:200]
                        print(f"  [tool_result] {tool_name} → (raw) {preview}")
                    if tc:
                        intermediate_steps.append((tc, msg.content))
        print("[Orchestrator Debug] ────────────────────────────────────────────\n")

        # Try to parse JSON from output with improved extraction
        plan = None
        
        # Method 1: Try markdown code fence with json tag
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(1))
                plan["resolution_method"] = "matched"
                print(f"[Orchestrator Debug] JSON parsed from ```json fence — resolution_method=matched")
            except json.JSONDecodeError as e:
                print(f"[Orchestrator Debug] JSON parse failed (method 1): {e}")
        
        # Method 2: Try markdown code fence without json tag
        if not plan:
            json_match = re.search(r'```\s*(\{.*?\})\s*```', output, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group(1))
                    plan["resolution_method"] = "matched"
                    print(f"[Orchestrator Debug] JSON parsed from ``` fence — resolution_method=matched")
                except json.JSONDecodeError as e:
                    print(f"[Orchestrator Debug] JSON parse failed (method 2): {e}")
        
        # Method 3: Try to find bare JSON object (greedy to capture nested structures)
        if not plan:
            # Find the first { and match everything until the last }
            start_idx = output.find('{')
            if start_idx != -1:
                # Count braces to find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(output)):
                    if output[i] == '{':
                        brace_count += 1
                    elif output[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = output[start_idx:end_idx]
                    try:
                        plan = json.loads(json_str)
                        plan["resolution_method"] = "matched"
                        print(f"[Orchestrator Debug] JSON parsed from bare object — resolution_method=matched")
                    except json.JSONDecodeError as e:
                        print(f"[Orchestrator Debug] JSON parse failed (method 3): {e}")
                        print(f"[Orchestrator Debug] Attempted JSON (first 500 chars): {json_str[:500]}")
        
        # Fallback to alternative strategies if JSON parsing failed
        if not plan:
            print(f"[Orchestrator Debug] All JSON parsing methods failed — extracting from tool steps")
            plan = self._create_basic_plan_from_steps(intermediate_steps)
            
            # If extraction also failed (no entities/predicates found), plan will be None
            if not plan:
                print(f"[Orchestrator Debug] Tool extraction failed — trying label group expansion")
                plan = self._try_label_group_expansion(user_query)

        # Ensure we have strategies
        if not plan or "strategies" not in plan or not plan["strategies"]:
            if plan:
                print(f"[Orchestrator Debug] Plan has no strategies — trying label group expansion")
            else:
                print(f"[Orchestrator Debug] No plan created — trying label group expansion")
            
            label_expansion_plan = self._try_label_group_expansion(user_query)
            if label_expansion_plan and label_expansion_plan.get("strategies"):
                plan = label_expansion_plan
                plan["resolution_method"] = "label_group_expansion"
                print(f"[Orchestrator Debug] Label group expansion successful — found {len(plan['strategies'])} strategies")
            else:
                print(f"[Orchestrator Debug] Label group expansion failed — falling back to top-connected")
                fallback = self._create_connectivity_fallback_plan()
                if plan:
                    plan["strategies"] = fallback["strategies"]
                    plan.setdefault("resolution_method", "fallback_top_connected")
                else:
                    plan = fallback

        # Limit to 5 strategies
        plan["strategies"] = plan["strategies"][:5]

        # Add metadata
        plan["raw_output"] = output
        plan["user_query"] = user_query

        return plan

    def _build_comprehensive_strategies(
        self,
        entity_ids: List[str],
        predicate_ids: List[str],
        community_ids: List[str],
        strategy_prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """Build comprehensive query strategies covering multiple approaches.

        Creates strategies for:
        1. Direct connections from entities
        2. Connected nodes for predicates
        3. Paths between entities

        Args:
            entity_ids: List of canonical entity IDs
            predicate_ids: List of canonical predicate IDs
            community_ids: List of community IDs
            strategy_prefix: Prefix for strategy names (e.g., "label_based_", "top_connected_")

        Returns:
            List of strategy dictionaries
        """
        strategies = []

        # Strategy 1: Direct triples involving entities
        if entity_ids:
            strategies.append({
                "name": f"{strategy_prefix}entity_direct",
                "description": "Direct triples and connections from the identified entities",
                "approach": "direct",
                "canonical_ids": {
                    "entities": entity_ids,
                    "predicates": [],
                },
                "community_ids": community_ids,
                "parameters": {"limit": 50},
            })

        # Strategy 2: Graph expansion from entities
        if entity_ids:
            strategies.append({
                "name": f"{strategy_prefix}entity_expansion",
                "description": "Expand 1-hop neighborhood relationships from the entities",
                "approach": "expansion",
                "canonical_ids": {
                    "entities": entity_ids,
                    "predicates": [],
                },
                "community_ids": community_ids,
                "parameters": {"depth": 1, "limit": 50},
            })

        # Strategy 3: Predicate-based connections
        if predicate_ids:
            strategies.append({
                "name": f"{strategy_prefix}predicate_connections",
                "description": "Find all connections using the identified relationship types",
                "approach": "direct",
                "canonical_ids": {
                    "entities": entity_ids if entity_ids else [],
                    "predicates": predicate_ids,
                },
                "community_ids": community_ids,
                "parameters": {"limit": 50},
            })

        # Strategy 4: Paths between entities (if we have 2+ entities)
        if entity_ids and len(entity_ids) >= 2:
            strategies.append({
                "name": f"{strategy_prefix}entity_paths",
                "description": "Find paths connecting the identified entities",
                "approach": "path",
                "canonical_ids": {
                    "entities": entity_ids,
                    "predicates": [],
                },
                "community_ids": community_ids,
                "parameters": {"depth": 3, "limit": 20},
            })

        # Strategy 5: Deeper expansion (2-hop) if we have few entities
        if entity_ids and len(entity_ids) <= 5:
            strategies.append({
                "name": f"{strategy_prefix}deep_expansion",
                "description": "Explore 2-hop relationships to find broader context",
                "approach": "expansion",
                "canonical_ids": {
                    "entities": entity_ids,
                    "predicates": [],
                },
                "community_ids": community_ids,
                "parameters": {"depth": 2, "limit": 30},
            })

        return strategies[:5]  # Limit to max 5 strategies

    def _build_community_exploration_strategy(
        self,
        community_ids: List[str],
        strategy_prefix: str = "",
    ) -> Dict[str, Any]:
        """Build a community exploration strategy that queries ALL relationships in a community.

        This strategy is used for broad community queries where the user wants to know
        everything about a community, not just specific entities.

        Args:
            community_ids: List of community IDs to explore
            strategy_prefix: Prefix for strategy name

        Returns:
            Strategy dictionary for community exploration
        """
        return {
            "name": f"{strategy_prefix}community_exploration",
            "description": f"Explore all relationships and entities within the community",
            "approach": "community_exploration",
            "canonical_ids": {
                "entities": [],
                "predicates": [],
            },
            "community_ids": community_ids,
            "parameters": {"limit": 100},
        }

    def _try_label_group_expansion(self, user_query: str) -> Optional[Dict[str, Any]]:
        """Try to find entities via label group expansion.

        This method:
        1. Uses LLM to identify the main label_group from the query
        2. Retrieves labels from that group via Qdrant
        3. Uses LLM to select the most relevant labels
        4. Queries entities with those labels
        5. Builds strategies using the found entities

        Args:
            user_query: The user's natural language question

        Returns:
            Plan dictionary with label-based strategies, or None if expansion fails
        """
        import json
        import yaml
        from pathlib import Path

        try:
            # Load label group config
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "label_group_config.yaml"
            
            with open(config_path) as f:
                label_groups = yaml.safe_load(f)
            
            # Create a formatted string of label groups for the LLM
            label_group_list = "\n".join([f"- {key}: {value}" for key, value in label_groups.items()])
            
            # Step 1: Ask LLM to identify relevant label group(s)
            identification_prompt = f"""Based on the user's question and the available label groups, identify the most relevant label group(s).

User Question: {user_query}

Available Label Groups:
{label_group_list}

Respond with ONLY a JSON object in this format:
{{
  "relevant_groups": ["GroupName1", "GroupName2"],
  "reasoning": "Brief explanation of why these groups are relevant"
}}

If multiple groups are relevant, list up to 3 in order of relevance. Use exact group names from the list above."""

            messages = [HumanMessage(content=identification_prompt)]
            response = self.llm.invoke(messages)
            
            # Parse label group identification
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response.content, re.DOTALL)
            
            if not json_match:
                print(f"[Label Expansion] Failed to parse label group identification")
                return None
            
            group_data = json.loads(json_match.group(1))
            relevant_groups = group_data.get("relevant_groups", [])
            
            if not relevant_groups:
                print(f"[Label Expansion] No relevant label groups identified")
                return None
            
            print(f"[Label Expansion] Identified label groups: {relevant_groups}")
            print(f"[Label Expansion] Reasoning: {group_data.get('reasoning', 'N/A')}")
            
            # Step 2: Retrieve labels from Qdrant for these groups
            all_labels = []
            for group in relevant_groups[:3]:  # Limit to top 3 groups
                try:
                    from kg_reasoning.agents.tools.qdrant_tools import _get_manager
                    manager = _get_manager()
                    labels = manager.get_labels_by_group(group, limit=50)
                    all_labels.extend(labels)
                    print(f"[Label Expansion] Found {len(labels)} labels in group '{group}'")
                except Exception as e:
                    print(f"[Label Expansion] Failed to retrieve labels for group '{group}': {e}")
            
            if not all_labels:
                print(f"[Label Expansion] No labels found for the identified groups")
                return None
            
            # Step 3: Ask LLM to select most relevant labels
            labels_summary = "\n".join([
                f"- {label['name']} (group: {label.get('group', 'N/A')}, type: {label.get('type', 'N/A')})"
                for label in all_labels[:100]  # Limit to avoid token overflow
            ])
            
            selection_prompt = f"""Based on the user's question, select the most relevant labels from the list below.

User Question: {user_query}

Available Labels:
{labels_summary}

Respond with ONLY a JSON object in this format:
{{
  "selected_labels": ["LabelName1", "LabelName2", "LabelName3"],
  "reasoning": "Brief explanation of why these labels are most relevant"
}}

Select up to 5 most relevant labels. Use exact label names from the list above."""

            messages = [HumanMessage(content=selection_prompt)]
            response = self.llm.invoke(messages)
            
            # Parse label selection
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', response.content, re.DOTALL)
            
            if not json_match:
                print(f"[Label Expansion] Failed to parse label selection")
                return None
            
            selection_data = json.loads(json_match.group(1))
            selected_labels = selection_data.get("selected_labels", [])
            
            if not selected_labels:
                print(f"[Label Expansion] No labels selected")
                return None
            
            print(f"[Label Expansion] Selected labels: {selected_labels}")
            print(f"[Label Expansion] Reasoning: {selection_data.get('reasoning', 'N/A')}")
            
            # Step 4: Query entities with those labels
            from kg_reasoning.agents.tools.qdrant_tools import _get_manager
            manager = _get_manager()
            entities = manager.get_entities_by_labels(selected_labels, limit=50)
            
            if not entities:
                print(f"[Label Expansion] No entities found for selected labels")
                return None
            
            print(f"[Label Expansion] Found {len(entities)} entities")
            
            # Extract canonical IDs
            entity_ids = [e["canonical_id"] for e in entities if e.get("canonical_id")]
            
            if not entity_ids:
                print(f"[Label Expansion] No valid entity IDs found")
                return None
            
            # Step 5: Build comprehensive strategies
            strategies = self._build_comprehensive_strategies(
                entity_ids=entity_ids,
                predicate_ids=[],
                community_ids=[],
                strategy_prefix="label_based_"
            )
            
            return {
                "entities_found": entity_ids,
                "predicates_found": [],
                "communities_identified": [],
                "label_groups_used": relevant_groups,
                "labels_selected": selected_labels,
                "resolution_method": "label_group_expansion",
                "strategies": strategies,
            }
        
        except Exception as e:
            print(f"[Label Expansion] Error during label group expansion: {e}")
            import traceback
            traceback.print_exc()
            return None

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

        strategies = self._build_comprehensive_strategies(
            entity_ids=top_ids,
            predicate_ids=[],
            community_ids=[],
            strategy_prefix="top_connected_"
        )

        return {
            "entities_found": top_ids,
            "predicates_found": [],
            "communities_identified": [],
            "resolution_method": "fallback_top_connected",
            "strategies": strategies,
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

        print(f"[Orchestrator Debug] Extracting from {len(intermediate_steps)} tool calls")

        # Extract information from tool outputs
        for action, observation in intermediate_steps:
            # Get tool name safely
            tool_name = ""
            if hasattr(action, 'tool'):
                tool_name = action.tool
            elif hasattr(action, 'get'):
                tool_name = action.get('name', '')
            elif isinstance(action, dict):
                tool_name = action.get('name', '')

            try:
                obs_data = json.loads(observation) if isinstance(observation, str) else observation

                # Extract entities from get_canonical_entities results
                if "canonical_entities" in tool_name or "get_canonical_entities" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, dict) and "canonical_id" in item:
                                entities.append(item["canonical_id"])
                                print(f"  [extracted] entity: {item.get('name')} ({item['canonical_id'][:8]}...)")

                # Extract predicates from get_canonical_predicates results
                elif "canonical_predicates" in tool_name or "get_canonical_predicates" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, dict) and "canonical_id" in item:
                                predicates.append(item["canonical_id"])
                                print(f"  [extracted] predicate: {item.get('name')} ({item['canonical_id'][:8]}...)")

                # Extract communities from get_community_metadata results
                elif "community_metadata" in tool_name or "get_community_metadata" in tool_name:
                    if isinstance(obs_data, list):
                        for item in obs_data:
                            if isinstance(item, dict) and "community_id" in item:
                                communities.append(item["community_id"])
                                print(f"  [extracted] community: {item['community_id']}")

                # Extract community IDs from get_community_ids_from_relationships
                elif "get_community_ids_from_relationships" in tool_name:
                    if isinstance(obs_data, list):
                        # This tool returns a list of strings
                        for item in obs_data:
                            if isinstance(item, str):
                                communities.append(item)
                                print(f"  [extracted] community: {item}")

            except Exception as e:
                print(f"  [extraction error] {tool_name}: {e}")
                pass

        entity_list = list(set(entities))
        predicate_list = list(set(predicates))
        community_list = list(set(communities))

        print(f"[Orchestrator Debug] Extracted: {len(entity_list)} entities, {len(predicate_list)} predicates, {len(community_list)} communities")

        if not entity_list and not predicate_list:
            print(f"[Orchestrator Debug] No entities/predicates extracted from tool calls — trying label expansion fallback")
            return None  # Signal to try label expansion

        strategies = self._build_comprehensive_strategies(
            entity_ids=entity_list,
            predicate_ids=predicate_list,
            community_ids=community_list,
            strategy_prefix="partial_match_"
        )

        return {
            "entities_found": entity_list,
            "predicates_found": predicate_list,
            "communities_identified": community_list,
            "resolution_method": "partial_match",
            "strategies": strategies,
        }
