"""Synthesizer agent for multi-agent reasoning system.

Reads query results from markdown files and synthesizes comprehensive answers
based on the user's original question.
"""

from typing import Any, Dict, List, Optional

from kg_extractor.utils.model_setup import REASONING_PROVIDER, SYNTHESIZER_MODEL
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
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

## How to Synthesize

1. **Read all results** using read_query_results (no filepath argument) to get every recent result file.
2. **Extract the facts** — ignore technical metadata; pull out names, events, quantities, dates, and connections that are relevant to the question.
3. **Answer directly** — open with a thorough, explanatory answer to the question. Go beyond a single sentence: explain the key drivers, dynamics, or mechanisms at play, and why they matter. Write 3–5 sentences.
4. **Add supporting detail** — elaborate with specifics drawn from the results (who, what, when, where, why, how).

## Output Format

- **Answer**: A thorough paragraph (3–5 sentences) that directly and fully answers the question — not just a summary statement, but an explanation of the underlying drivers, relationships, and significance.
- **Details**: Supporting facts and context in natural prose or a simple bullet list.

Do not include sections titled "Limitations", "Data Sources", "Strategies", "Confidence Level", or any other technical meta-section.

## Language Rule

**Always respond in the same language as the user's question.** If the question is in Thai, answer entirely in Thai. If in English, answer in English. Do not switch languages."""


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
            llm_provider: LLM provider (currently only "openai" supported)
            llm_model: Model name (use capable model for synthesis)
            temperature: LLM temperature (higher for more creative synthesis)
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
            "Please read all available query results and write a plain-language answer to the question above."
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

        for step in intermediate_steps:
            if hasattr(step[0], "name"):
                if "read_query_results" in step[0].name:
                    files_read += 1
                if "list_query_results" in step[0].name:
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
