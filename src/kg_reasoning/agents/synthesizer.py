"""Synthesizer agent for multi-agent reasoning system.

Reads query results from markdown files and synthesizes comprehensive answers
based on the user's original question.
"""

from typing import Any, Dict, List, Optional

from kg_extractor.utils.model_setup import REASONING_PROVIDER, SYNTHESIZER_MODEL, get_reasoning_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from kg_reasoning.agents.tools.markdown_tools import (
    read_query_results,
    list_query_results,
)


SYNTHESIZER_SYSTEM_PROMPT = """You are a Knowledge Answer Synthesizer. Your job is to read raw query results from a knowledge graph and produce a clear, natural-language answer to the user's question.

## Core Rule

**Write for a human reader who knows nothing about graphs.**
- Never mention nodes, edges, relationships, predicates, canonical IDs, communities, Cypher, or any other graph/database concept.
- Translate everything into plain subject-matter language. If a result says Entity A has a relationship "causes" to Entity B, write "A causes B."
- Focus entirely on the real-world meaning of the data.

## CRITICAL: No Results = No Answer

**If the query results show "No results found" or 0 records, you MUST state that no information is available in the knowledge graph.**
- Do NOT make up or infer information.
- Do NOT provide generic answers based on the question topic.
- Do NOT use any knowledge outside of the query results.
- Simply state clearly that the knowledge graph has no information about the requested topic.

## How to Synthesize

1. **FIRST: Call read_query_results tool** with an empty string for filepath parameter: read_query_results(filepath="") to read ALL recent result files.
2. **Check if there are any results** — if all files show "No results found" or 0 records, immediately state that no information is available and stop.
3. **Extract the facts** (only if results exist) — ignore technical metadata; pull out names, events, quantities, dates, and connections relevant to the question.
4. **Determine the answer type** — decide which mode applies based on what the question asks for:
   - **Factual**: the answer already exists in the results (e.g., "what happened", "who is involved", "what are the issues"). Stay strictly within the retrieved context; do not introduce claims beyond what the results establish.
   - **Generative**: the question asks for something not stored in the graph — solutions, recommendations, action plans, diagnoses, or explanations of cause. Use the retrieved context as the mandatory foundation (the problems, constraints, and entities are defined by the results), then apply your knowledge to generate a relevant, grounded response. Do not fabricate the problem — only the solution may come from your knowledge.
5. **Stay concise** — include only what directly answers the question. Omit tangents, background, or elaborations not connected to the retrieved context.

**CRITICAL**: You MUST call the read_query_results tool. Do not try to answer without reading the query results first.

## Output Format

**When results exist:**
- **Answer**: A focused response that directly addresses the question. For factual questions, every claim traces back to the retrieved context. For generative questions, the problems and constraints come from the context; solutions and recommendations may draw on your broader knowledge.
- **Details**: Supporting facts from the results, in natural prose or a concise bullet list. For generative responses, clearly distinguish what the context established from what you are proposing.

**When no results exist:**
- **Answer**: A clear statement that the knowledge graph has no information about the requested topic.

Do not include sections titled "Limitations", "Data Sources", "Strategies", "Confidence Level", or any other meta-section. Do not pad the response with caveats, general background, or filler.

## Language Rule

**Always respond in the same language as the user's question.** If the question is in Thai, answer entirely in Thai. If in English, answer in English. Do not switch languages.
"""


class SynthesizerAgent:
    """Synthesizer agent for generating final answers."""

    def __init__(
        self,
        llm_provider: str = REASONING_PROVIDER,
        llm_model: str = SYNTHESIZER_MODEL,
        temperature: float = 0.3,
    ):
        """Initialize the synthesizer agent.

        Args:
            llm_provider: LLM provider (supports: openai, openrouter, groq, nvidia)
            llm_model: Model name (use capable model for synthesis)
            temperature: LLM temperature (higher for more creative synthesis)
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        # Initialize LLM using helper function that supports multiple providers
        self.llm = get_reasoning_llm(model=llm_model, temperature=temperature)

        # Setup tools
        self.tools = [
            read_query_results,
            list_query_results,
        ]

        # Create agent using langgraph
        self.agent = create_react_agent(self.llm, self.tools)

    def synthesize_answer(
        self,
        user_query: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Synthesize final answer from query results.

        Args:
            user_query: Original user question
            strategies: Optional list of strategies that were executed

        Returns:
            Dictionary with synthesized answer and metadata
        """
        instruction = f"""{SYNTHESIZER_SYSTEM_PROMPT}

**Question:** {user_query}

"""

        if strategies:
            instruction += (
                f"*(The system ran {len(strategies)} background queries to gather relevant data.)*\n\n"
            )

        instruction += (
            "Please call the read_query_results tool with filepath=\"\" to read all available query results, then write a plain-language answer to the question above."
        )

        # Execute agent
        messages = [HumanMessage(content=instruction)]
        result = self.agent.invoke({"messages": messages})

        # Extract answer from messages — only use the final AIMessage (not tool
        # calls or the initial HumanMessage that contains the full system prompt)
        answer = ""
        intermediate_steps = []

        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, AIMessage):
                    # A final AI reply has content but no pending tool calls
                    if msg.content and not getattr(msg, "tool_calls", None):
                        answer = msg.content  # keep overwriting; last one is the answer
                    if getattr(msg, "tool_calls", None):
                        for tool_call in msg.tool_calls:
                            intermediate_steps.append((tool_call, None))

        # Count how many result files were read
        files_read = 0
        results_analyzed = 0

        # Check if tools were actually called
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name", "")
                            if "read_query_results" in tool_name:
                                files_read += 1
                            elif "list_query_results" in tool_name:
                                files_read += 1

        return {
            "answer": answer.strip(),
            "user_query": user_query,
            "files_read": files_read,
            "results_analyzed": results_analyzed,
            "synthesis_quality": self._assess_quality(answer),
        }

    def _assess_quality(self, answer: str) -> str:
        """Assess the quality of synthesized answer.

        Args:
            answer: Synthesized answer text

        Returns:
            Quality assessment string
        """
        # Simple heuristic-based quality assessment
        if not answer or len(answer) < 50:
            return "poor"

        # Check for structure indicators
        has_evidence = any(keyword in answer.lower() for keyword in [
            "according to", "based on", "found in", "shows that", "indicates"
        ])

        has_detail = len(answer) > 200

        has_structure = any(marker in answer for marker in [
            "**", "##", "1.", "2.", "-", "•"
        ])

        quality_score = sum([has_evidence, has_detail, has_structure])

        if quality_score >= 3:
            return "high"
        elif quality_score >= 2:
            return "medium"
        else:
            return "basic"
