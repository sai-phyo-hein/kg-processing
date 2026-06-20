"""Synthesizer agent for multi-agent reasoning system.

Reads the aggregator's single community-grouped markdown file and
synthesizes a full answer to the user's original question.

INPUT FORMAT (from aggregator.py):
The aggregator no longer writes flat JSON-block results. It writes one file
grouped by community_id, with explicit boundary markers around each section:

    <!-- COMMUNITY_START: {community_id} -->
    # {community_id}
    ## relationships
    {subject} -> ({predicate}) -> {object} | {attribute} | {causal} | {temporal}
    {raw name} -> (linked) -> {canonical name}
    ## relevant content (references for relationships)
    {chunk text}
    <!-- COMMUNITY_END: {community_id} -->

This file is read directly in Python (deterministic I/O, no LLM round trip)
and inlined into LLM call(s) — no ReAct loop, no tool the model has to
remember to invoke.

ANSWER PHILOSOPHY (changed from the original "stay concise" default):
The synthesizer must answer FULLY, not summarize. Reasoning over the
retrieved relationships and chunk text, and proposing solutions/recommendations
grounded in that content, is a standing capability — not a special mode
gated behind detecting a "generative" question type. See
SYNTHESIZER_SYSTEM_PROMPT for the full behavior.

SCALING — ONE-SHOT VS MAP-REDUCE (NEW):
A single LLM call reading every community's full relationships+chunks does
not scale past a handful of communities — 10 communities' worth of capped
content can still add up to tens of thousands of characters, well past what
any one call should be asked to reason over carefully in one pass. The
threshold is len(community_ids) >= 3:
  - BELOW 3 communities: one-shot — the existing single LLM call reads the
    whole aggregated file directly. This is the common case (single-village
    questions, two-village comparisons) and keeps the simplicity/cost of one
    call, with full cross-community reasoning available in one pass.
  - AT OR ABOVE 3 communities: map-reduce — aggregator.split_by_community()
    splits the file into per-community sections (using the explicit
    COMMUNITY_START/COMMUNITY_END markers, not heading-text parsing), each
    section is summarized by its OWN LLM call (run in parallel, same
    ThreadPoolExecutor pattern the rest of this codebase already uses for
    independent I/O-bound work), and a final LLM call combines those
    per-community summaries (not the raw sections) into the full answer.
    This scales to arbitrarily many communities since each summarization
    call's input size is bounded by the aggregator's own per-community caps
    (40 relationships, 8000 chars of chunk text — see aggregator.py),
    regardless of how many communities the query spans.

NO-RESULTS DETECTION (fixed):
The aggregator returns {"filepath": None, "result_count": 0} when it found
nothing at all — it never writes a file in that case. The synthesizer must
treat aggregated_filepath is None / result_count == 0 as the no-results
signal directly, rather than falling back to reading whatever recent files
happen to sit in the output directory (which could be stale results from an
earlier, unrelated query) or string-matching content for phrases the new
format doesn't even produce.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from kg_extractor.utils.model_setup import REASONING_PROVIDER, SYNTHESIZER_MODEL, get_reasoning_llm
from langchain_core.messages import HumanMessage, SystemMessage

from kg_reasoning.agents.aggregator import split_by_community


# Community count at/above which the map-reduce path replaces the one-shot
# path. Kept as a module constant rather than buried in a method body so
# the threshold is easy to find and tune in one place.
MAP_REDUCE_COMMUNITY_THRESHOLD = 3

_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")


def _detect_response_language(user_query: str) -> str:
    """Detect which language the final answer must be written in, FROM THE
    QUESTION ALONE — not inferred separately by each LLM call from a
    question line buried among walls of retrieved content.

    Why this exists: in the map-reduce path, retrieved chunk/relationship
    content is predominantly Thai regardless of what language the user
    asked in. Each per-community summarization call only sees the
    question once, as a single short line, surrounded by a much larger
    volume of Thai source text — and a plain "respond in the same
    language as the question" instruction can lose to that volume
    imbalance, producing a Thai summary even for an English question.
    Once that happens, the combine step receives already-Thai summaries
    as its dominant input and is liable to mirror THAT language instead
    of re-deriving the right one from the original question.

    The fix is to settle the language ONCE, here, from the question text
    only (never from retrieved content), and hand every downstream prompt
    an explicit instruction ("respond in English." / "respond in Thai.")
    rather than leaving each call to infer it under different and
    unequal amounts of competing-language context.

    Detection is deliberately narrow — Thai-script presence via Unicode
    range \\u0E00-\\u0E7F — rather than a general-purpose language
    detector, since this pipeline's only two observed question languages
    are Thai and English. Any Thai character anywhere in the question
    is treated as a Thai question, even mixed with English words, since
    that's the more common real pattern (e.g. an English question naming
    a Thai place name) and Thai should win that mix per the user's
    evident intent to engage in Thai.

    Returns "Thai" or "English" — never a graph/pipeline-internal value,
    since this is inserted directly into a human-readable instruction.
    """
    if _THAI_CHAR_RE.search(user_query or ""):
        return "Thai"
    return "English"




SYNTHESIZER_SYSTEM_PROMPT = """You are a Knowledge Answer Synthesizer. Your job is to read retrieved knowledge-graph content and produce a FULL, complete answer to the user's question — not a summary of it.

## Core Rule: Write For A Human Reader

- Never mention nodes, edges, relationships, predicates, canonical IDs, communities, Cypher, or any other graph/database concept by name.
- Translate everything into plain subject-matter language. If the content says "A -> (CAUSES) -> B", write "A causes B."
- Focus entirely on the real-world meaning of the data.

## Understanding The Input Format

The content below is organized by community (village/group), each with two parts:
- **relationships**: lines shaped `X -> (relation) -> Y | extra: value | extra: value`. The
  part after any `|` is additional detail about that specific relationship — a strength/weight,
  a time period, or some other attribute — when the underlying data actually had one. Use these
  details as real evidence in your answer (e.g. "this had a notably high impact" if a weight
  value supports that), not as something to mention mechanically.
- A second kind of line shaped `raw term -> (linked) -> canonical term` (sometimes `canonical
  term [English translation]` when the data has one) tells you two or three different surface
  forms refer to the same real-world thing (e.g. a Thai phrase and its standardized form, plus
  an English translation in brackets when available). Treat all forms as the same entity when
  answering — never present them as different things, and never expose the words "linked" or
  "canonical" to the reader. If a bracketed English form is present, prefer using it (or a
  natural phrasing of it) when writing in English, since it's the curated translation.
  **Use this link to connect source text to graph structure**: the raw term is the exact
  surface form you'll actually encounter when reading the relevant content passages below (it's
  how that name appears verbatim in the source text), while the canonical term is what's used in
  the relationship lines above. When a passage in relevant content mentions the raw term, that
  passage is providing supporting detail for whatever relationships involve the matching
  canonical term — connect the two when reasoning, e.g. if relevant content describes something
  about the raw term and a relationship line says the canonical term causes some outcome, treat
  that passage as evidence for that specific relationship, not as a disconnected fact.
- **relevant content**: actual source passages backing the relationships above. This is your
  richest evidence — read it for nuance, qualifiers, and detail the relationship lines alone
  compress away (exact wording, numbers, reasoning given by the original speaker, etc.).

## CRITICAL: No Results = No Answer

**If you are told no information was retrieved, you MUST state plainly that the knowledge graph has no information about the requested topic.**
- Do NOT make up or infer information.
- Do NOT provide generic answers based on the question topic.
- Do NOT use outside knowledge to fill the gap.
- This rule is about MISSING information only — it does not restrict you from reasoning over information that DOES exist (see below).

## How To Synthesize — Answer Fully

1. **Check whether anything was retrieved.** If not, apply the No Results rule above and stop.
2. **Read everything**: relationships, their attributes, the raw/canonical links, and the
   relevant-content passages. The passages often contain detail the relationship lines alone
   lose — use them, don't just paraphrase the relationship lines.
3. **Answer completely, not minimally.** Include every fact, connection, and nuance from the
   retrieved content that bears on the question — across every community section that's
   relevant, not just the first one. A full answer is the default; do not compress it into a
   short summary or cut it down to "just the headline fact" when the retrieved content supports
   more.
4. **Reason and propose solutions when it's warranted.** You are not limited to restating facts.
   When the question asks for, or would benefit from, explanation, diagnosis, or a course of
   action — "why does this happen", "what should be done", "how could this be improved" — use
   the retrieved relationships and content as the grounded foundation (the problems, people,
   and constraints are defined by what was retrieved; do not invent a problem that isn't there),
   then apply your own reasoning and knowledge to explain causes or propose concrete, specific
   solutions. This is a normal, expected part of answering — not a special mode you need
   permission to enter.
5. **Stay grounded.** Every claim about WHAT EXISTS or WHAT HAPPENED must come from the
   retrieved content. Your own reasoning is for connecting, explaining, and extending those
   facts toward an answer or solution — not for asserting new facts that contradict or aren't
   present in what was retrieved.

## Output Format

- Write the complete answer — as long as it needs to be to cover the relevant retrieved
  content fully. Do not artificially shorten it.
- Structure it for readability (headers, lists) when the content has multiple parts worth
  separating, but structure is in service of completeness, not a substitute for it.
- Do not include sections titled "Limitations", "Data Sources", "Strategies", "Confidence
  Level", or other meta-commentary about the retrieval process itself.
- When no information was found, the answer is simply that clear statement — nothing more.

## Language Rule

**Always respond in the same language as the user's question.** If the question is in Thai, answer entirely in Thai. If in English, answer in English. Do not switch languages. The retrieved content is very often in a different language than the question (commonly Thai source content for an English question) — that is normal, and is NOT a signal to answer in the content's language; the question's language always wins.
"""


COMMUNITY_SUMMARY_SYSTEM_PROMPT = """You are summarizing ONE community's (village/group's) retrieved knowledge-graph content, as one step of a larger multi-community answer. Your summary will be combined with other communities' summaries later — you are not writing the final answer, so do not address the user directly or write introductions/conclusions framed as a complete response.

## Your Job

Read this ONE community's relationships, raw->canonical name links, and reference content,
then write a dense, complete summary of everything in it that's relevant to the user's
question. Treat this as compression, not selection — the goal is to preserve every fact,
name, number, and connection that bears on the question, just stated efficiently, not to
decide which facts are "important enough" to keep and discard the rest.

The raw->canonical links matter for connecting reference content to relationships: a raw
name is the exact surface form that appears verbatim in the reference content passages, while
the matching canonical name is what the relationship lines use. When a passage mentions a raw
name, treat it as supporting detail for whatever relationships involve that same name's
canonical form — don't summarize the passage and the relationship as if they were unconnected.

## What To Preserve

- Every named entity (person, group, organization, place) and what role/relationship it has
- Every relationship's substance (what connects to what, and how) — translate graph jargon
  (predicates, canonical IDs) into plain language as you go, the same as a full answer would
- Any attribute, strength, or timing detail attached to a relationship (the "| weight: ...",
  "| temporal_link: ..." style annotations) — these are real signal, not metadata to drop
- Specific facts and figures from the reference content (the actual source passages) — these
  often contain nuance, exact wording, or numbers the relationship lines alone compress away
- If this community has little or no relevant content, say that explicitly and briefly —
  "no relevant information found for this community" is itself a useful, complete summary,
  not a failure to summarize

## What NOT To Do

- Do not write a conclusion, recommendation, or comparison to other communities — you don't
  have visibility into other communities' content, and that synthesis happens in a later step
- Do not pad with generic framing ("This community shows...", "In summary..."); every line
  should carry information
- Do not mention graph/database mechanics (nodes, edges, canonical IDs, "the aggregator")

## Language Rule

**Write this summary in the language the calling instruction specifies — not necessarily
the language of the retrieved content above.** The content you're summarizing is very
often in a different language than the question (e.g. Thai source content for an English
question) — that is normal and expected, and is NOT a signal to summarize in the content's
language. Follow the explicit language instruction given to you for this call.
"""


COMBINE_SUMMARIES_SYSTEM_PROMPT = """You are a Knowledge Answer Synthesizer. You will receive per-community summaries — each one already a dense compression of that community's retrieved knowledge-graph content — and must combine them into ONE full, complete answer to the user's original question.

## Core Rule

Write for a human reader who knows nothing about graphs or about how these summaries were
produced. Never mention "summaries", "communities" as a technical concept, canonical IDs, or
any retrieval/pipeline mechanics — refer to villages/groups by their actual names/identities
as the content does.

## CRITICAL: No Results = No Answer

If every per-community summary states no relevant information was found, you MUST state
plainly that the knowledge graph has no information about the requested topic. Do not invent
information to fill the gap.

## How To Combine

1. **Read every summary fully** before writing anything — the question may need information
   from several communities, not just the first one.
2. **Answer completely, not minimally.** Include every fact and connection from the summaries
   that bears on the question, across every relevant community — a full answer is the default.
3. **If the question is comparative** ("compare X and Y", "how do they differ"), structure the
   answer to actually compare — explicitly note similarities AND differences across
   communities, not just a list of separate per-community descriptions stitched together.
   A community with little/no information is itself part of the comparison ("village 4 has no
   recorded health program, unlike villages 1-3") — say so rather than omitting it.
4. **Reason and propose solutions when warranted** — the same as a normal answer would: if the
   question asks why something happens or what should be done, use the combined retrieved
   content as the grounded foundation, then apply your own reasoning to explain or propose,
   without inventing problems that aren't present in the summaries.
5. **Stay grounded.** Every claim about what exists or happened must trace back to the
   summaries. Your own reasoning connects and extends those facts; it doesn't replace them.

## Output Format

- Write the complete answer — as long as it needs to be to cover everything relevant across
  all communities. Do not artificially shorten it.
- Structure it for readability (headers, lists, a comparison table if that suits the
  question) — multi-community answers usually benefit from clear structure.
- Do not include sections titled "Limitations", "Data Sources", "Strategies", "Confidence
  Level", or other meta-commentary about the retrieval process itself.

## Language Rule

**Always respond in the same language as the user's original question, exactly as stated in the explicit language instruction given to you for this call.** Per-community summaries should already be in that language, but if any individual summary drifted into a different one, do not mirror it — the question's language always wins over a summary's.
"""


class SynthesizerAgent:
    """Synthesizer agent for generating final answers via a single direct LLM call."""

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
        self.llm = get_reasoning_llm(model=llm_model, temperature=temperature)
        # No tools, no create_react_agent — this is now a plain chat call.

    def synthesize_answer(
        self,
        user_query: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        aggregated_filepath: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize a full answer from the aggregator's community-grouped file.

        Branches into one-shot vs map-reduce based on how many communities
        the file actually contains (see MAP_REDUCE_COMMUNITY_THRESHOLD and
        the module docstring) — community count is derived directly from
        the file's own COMMUNITY_START/COMMUNITY_END markers via
        aggregator.split_by_community, not passed in separately, so this
        stays self-consistent with whatever the aggregator actually wrote
        rather than trusting a count from elsewhere in the pipeline that
        could disagree with the file's real content.

        Args:
            user_query: Original user question
            strategies: Optional list of strategies that were executed (for metadata only)
            aggregated_filepath: Path to the community-grouped markdown file
                produced by the aggregator step. The aggregator returns None
                here specifically when it found zero results across every
                worker output (see aggregator.aggregate_results) — that is
                the authoritative no-results signal, not a string pattern
                to search for in file content, and not a cue to fall back
                to reading unrelated recent files from a previous query.

        Returns:
            Dictionary with synthesized answer and metadata
        """
        if not aggregated_filepath:
            # The aggregator already determined there is nothing to
            # synthesize from (result_count == 0) — do not read any file,
            # since "recent files" could belong to an earlier, unrelated
            # query and would misrepresent this query's actual result.
            content = ""
            files_read = 0
        else:
            content = self._read_aggregated_file(aggregated_filepath)
            files_read = 1

        no_results = not aggregated_filepath or content.strip() == ""

        if no_results:
            return self._no_results_response(user_query, files_read)

        sections = split_by_community(content)

        if len(sections) >= MAP_REDUCE_COMMUNITY_THRESHOLD:
            print(f"  🗺️  {len(sections)} communities >= threshold "
                  f"({MAP_REDUCE_COMMUNITY_THRESHOLD}) — using map-reduce synthesis")
            answer = self._synthesize_map_reduce(user_query, sections)
        else:
            print(f"  📄 {len(sections) or 1} community section(s) — using one-shot synthesis")
            answer = self._synthesize_one_shot(user_query, content)

        results_analyzed = self._count_results(content)

        return {
            "answer": answer,
            "user_query": user_query,
            "files_read": files_read,
            "results_analyzed": results_analyzed,
            "synthesis_quality": self._assess_quality(answer, no_results),
        }

    @staticmethod
    def _read_aggregated_file(filepath: str) -> str:
        """Read the aggregator's file directly, with NO character cap.

        markdown_tools.MarkdownToolsManager.read_query_results defaults to
        max_chars_per_file=8000 — that default exists for its OTHER use
        case (combining up to 10 unrelated recent files into one bounded
        context when no specific filepath is given), and silently
        truncates ANY single file past 8000 chars, appending only a small
        "*[Content truncated for length]*" marker easy to miss in a large
        prompt. A real aggregated file with 2+ communities routinely
        exceeds 8000 chars (the aggregator's own per-community caps alone
        allow up to ~48000 chars for 2 communities: 40 relationships +
        8000 chars of chunk text each) — going through that method here
        silently dropped every community after the first ~8000 characters,
        which is exactly what caused a real "Village 1 has no data" answer
        when Village 1's section was simply never read at all.

        The aggregator already controls this file's total size
        deliberately, via its own per-community relationship/chunk caps
        (see aggregator.py) — there's no reason for the synthesizer to
        impose an unrelated, smaller cap on top of that on its way in.
        """
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found: {filepath}"
        return path.read_text(encoding="utf-8")

    def _no_results_response(self, user_query: str, files_read: int) -> Dict[str, Any]:
        """Build the no-results response.

        Still makes one LLM call — not to decide WHETHER there's an
        answer (that's already certain: there isn't), but so the fixed
        "no information found" message comes back in the user's own
        question language. The language is detected once via
        _detect_response_language and stated explicitly, rather than left
        for the model to infer from the question text alone.
        """
        language = _detect_response_language(user_query)
        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"**Question:** {user_query}\n\n"
                f"**Retrieved knowledge graph results:** (none — nothing was "
                f"found for this question)\n\n"
                f"Follow the No Results rule in your instructions exactly. "
                f"Respond in {language}."
            )),
        ]
        response = self.llm.invoke(messages)
        answer = (response.content or "").strip()
        return {
            "answer": answer,
            "user_query": user_query,
            "files_read": files_read,
            "results_analyzed": 0,
            "synthesis_quality": self._assess_quality(answer, no_results=True),
        }

    def _synthesize_one_shot(self, user_query: str, content: str) -> str:
        """Original single-call path: read the whole aggregated file
        directly. Used when community count is below
        MAP_REDUCE_COMMUNITY_THRESHOLD.
        """
        language = _detect_response_language(user_query)
        user_message = f"""**Question:** {user_query}

**Retrieved knowledge graph results:**

{content}

Write a full, complete answer to the question above, following the system instructions exactly. Respond in {language}, regardless of what language the retrieved results above are written in."""

        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        return (response.content or "").strip()

    def _synthesize_map_reduce(
        self, user_query: str, sections: List[Any]
    ) -> str:
        """Map-reduce path: summarize each community's section in its own
        parallel LLM call (map), then combine those summaries into the
        full answer with one final LLM call (reduce).

        Each per-community call's input is already bounded by the
        aggregator's own per-community caps (40 relationships, 8000 chars
        of chunk text), so this scales to arbitrarily many communities —
        the combine step's input size grows with community COUNT, not with
        each community's raw content volume, since it only ever sees
        already-compressed summaries.

        Language is detected ONCE here, from user_query alone (see
        _detect_response_language), and passed explicitly into both the
        summarize and combine steps below — not re-inferred separately by
        each call from a question line that's vastly outweighed by Thai
        source content in the summarize step, and by Thai-or-not summaries
        in the combine step. Settling it once here means the combine step
        is never in a position to mirror a summary's language instead of
        the question's, because every summary was already produced in the
        target language to begin with.

        Args:
            user_query: Original user question
            sections: [(community_id, section_text), ...] from
                aggregator.split_by_community

        Returns:
            The final combined answer string
        """
        language = _detect_response_language(user_query)
        summaries = self._summarize_communities_parallel(user_query, sections, language)
        return self._combine_summaries(user_query, summaries, language)

    def _summarize_communities_parallel(
        self, user_query: str, sections: List[Any], language: str
    ) -> List[Any]:
        """Run one LLM call per community section, in parallel — same
        ThreadPoolExecutor pattern used elsewhere in this codebase
        (preprocessor.py's canonical lookups, workers_node's spec
        execution) for independent I/O-bound work. LLM calls are
        network-bound, not CPU-bound, so a thread pool (not a process
        pool) is the right tool here.

        Returns [(community_id, summary_text), ...] in the SAME order as
        `sections` was given — ThreadPoolExecutor.map preserves input
        order in its output, so the combine step sees communities in a
        stable, predictable order (matching the aggregator's own
        relevance-based ordering) rather than whatever order calls happened
        to finish in.

        Args:
            user_query: Original user question
            sections: [(community_id, section_text), ...]
            language: Pre-detected target language ("Thai"/"English"),
                stated explicitly so this call doesn't have to infer it
                from a one-line question surrounded by a much larger
                volume of (typically Thai) source content.
        """
        def _summarize_one(section: Any) -> Any:
            community_id, section_text = section
            messages = [
                SystemMessage(content=COMMUNITY_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"**User's Question:** {user_query}\n\n"
                    f"**This community's retrieved content:**\n\n{section_text}\n\n"
                    f"Summarize everything in this community's content that's relevant "
                    f"to the question, following your instructions exactly. Write the "
                    f"summary in {language}, regardless of what language the retrieved "
                    f"content above is written in."
                )),
            ]
            response = self.llm.invoke(messages)
            return (community_id, (response.content or "").strip())

        with ThreadPoolExecutor(max_workers=min(len(sections), 8)) as pool:
            return list(pool.map(_summarize_one, sections))

    def _combine_summaries(self, user_query: str, summaries: List[Any], language: str) -> str:
        """Final reduce step: combine all per-community summaries into one
        full answer. Single LLM call — its input is the SUM of summaries,
        not raw content, so it stays bounded regardless of community count.

        Args:
            user_query: Original user question
            summaries: [(community_id, summary_text), ...] from
                _summarize_communities_parallel — already produced in
                `language` (see that method), so this step's input is
                already in the right language, not a separate inference.
            language: Pre-detected target language, restated explicitly
                here too as a backstop in case any individual summary
                didn't fully comply.
        """
        summary_block = "\n\n".join(
            f"### Community: {cid}\n{summary}" for cid, summary in summaries
        )

        messages = [
            SystemMessage(content=COMBINE_SUMMARIES_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"**Question:** {user_query}\n\n"
                f"**Per-community summaries:**\n\n{summary_block}\n\n"
                f"Combine these into one full, complete answer to the question above, "
                f"following the system instructions exactly. Respond in {language}."
            )),
        ]
        response = self.llm.invoke(messages)
        return (response.content or "").strip()

    @staticmethod
    def _count_results(content: str) -> int:
        """Count retrieved items in the aggregator's community-grouped format.

        The old flat-JSON format had one "### Result N" heading per item,
        making the count trivial. The new format has no such heading — it
        groups relationship lines under "## relationships" per community.
        Count relationship lines instead: every non-blank line under a
        "## relationships" section that isn't itself a raw->canonical link
        line (those describe naming, not a retrieved relationship) and
        isn't a sub-heading.
        """
        if not content.strip():
            return 0

        count = 0
        in_relationships = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## relationships"):
                in_relationships = True
                continue
            if stripped.startswith("## ") or stripped.startswith("# "):
                in_relationships = False
                continue
            if in_relationships and stripped and "-> (linked) ->" not in stripped:
                count += 1
        return count

    def _assess_quality(self, answer: str, no_results: bool) -> str:
        """Assess the quality of synthesized answer.

        Args:
            answer: Synthesized answer text
            no_results: Whether the underlying retrieval found nothing

        Returns:
            Quality assessment string
        """
        if no_results:
            # Correctly reporting "no information available" is the right
            # behavior, not a poor answer — don't penalize it the way a
            # short, weak answer over real results would be penalized.
            return "no_data"

        if not answer or len(answer) < 50:
            return "poor"

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