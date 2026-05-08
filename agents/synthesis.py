# agents/synthesis.py

from typing import List, Set

from agents.state import (
    AgentState,
    EvidenceRecord,
    BridgeCandidate,
)

from core.rag_tools import (
    RetrievedChunk,
    retrieve_chunks,
    generate_grounded,
)

from core.llm import invoke

# =========================
# Prompt Template
# =========================

SYNTHESIS_PROMPT = """
You are a quantum machine learning research synthesis system.

You must produce a structured epistemic analysis using ONLY the supplied context.

STRICT FORMAT:

ESTABLISHED
-----------
- Facts with verdict="supported" and confidence > 0.7
- Cite paper sources inline using [Source: ...]
- Only include well-supported claims

PROBABLE BRIDGES
----------------
- Describe speculative or supported cross-domain bridges
- Explicitly name the bridge_concept
- Include confidence scores
- Keep claims cautious and evidence-based

OPEN QUESTION
-------------
- Frame the single highest-novelty bridge as a research hypothesis
- Format:
  "It remains unexplored whether X and Y interact through Z.
   Evidence suggests this connection is N% probable
   based on M supporting sources."

LIMITATIONS
-----------
- Include contradicted evidence if present
- Mention corpus limitations
- Mention degraded execution if applicable

Requirements:
- Maximum 800 words
- Do NOT invent citations
- Use ONLY provided source metadata
- Be scientifically cautious
- If evidence is weak, explicitly say so

Original Question:
{question}

Grounded Base Answer:
{base_answer}

Structured Evidence:
{evidence_summary}

Bridge Candidates:
{bridge_summary}

System Notes:
{system_notes}
"""

# =========================
# Helpers
# =========================

def unique_chunks_from_evidence(
    evidence_records: List[EvidenceRecord]
) -> List[RetrievedChunk]:

    seen = set()
    collected = []

    for record in evidence_records:

        if record.verdict == "contradicted":
            continue

        for chunk in record.supporting_chunks:

            key = (chunk.source, chunk.content[:100])

            if key not in seen:
                seen.add(key)
                collected.append(chunk)

    return collected


def build_evidence_summary(
    evidence_records: List[EvidenceRecord]
) -> str:

    if not evidence_records:
        return "No bridge evidence available."

    lines = []

    for i, record in enumerate(evidence_records, start=1):

        sources = sorted({
            c.source for c in record.supporting_chunks
        })

        lines.append(
            f"""
Evidence #{i}
Candidate:
  {record.candidate.triple_a.subject}
  ->
  {record.candidate.bridge_concept}
  ->
  {record.candidate.triple_b.subject}

Verdict: {record.verdict}
Confidence: {record.confidence:.2f}

Supporting Sources:
{", ".join(sources) if sources else "None"}
""".strip()
        )

    return "\n\n".join(lines)


def build_bridge_summary(
    bridge_candidates: List[BridgeCandidate]
) -> str:

    if not bridge_candidates:
        return "No bridge candidates identified."

    lines = []

    for i, candidate in enumerate(bridge_candidates, start=1):

        lines.append(
            f"""
Bridge #{i}
A: {candidate.triple_a.subject}
B: {candidate.triple_b.subject}
Bridge Concept: {candidate.bridge_concept}
Novelty Score: {candidate.novelty_score:.2f}
Co-occurrence Count: {candidate.co_occurrence_count}
""".strip()
        )

    return "\n\n".join(lines)


def build_system_notes(state: AgentState) -> str:

    notes = []

    if state.get("degraded_mode", False):

        notes.append(
            "WARNING: System executed in degraded mode."
        )

        errors = state.get("errors", [])

        if errors:
            notes.append(
                "Errors encountered: "
                + "; ".join(errors)
            )

    notes.append(
        "This analysis is bounded by the ingested corpus "
        "of ~500 QML papers."
    )

    return "\n".join(notes)


# =========================
# Main Node
# =========================

def synthesize(state: AgentState) -> AgentState:

    original_query = state.get("original_query", "")
    evidence_records = state.get("evidence", [])
    bridge_candidates = state.get("bridge_candidates", [])

    llm_calls_used = state.get("llm_calls_used", 0)

    # =================================
    # Fallback Mode
    # =================================

    if not evidence_records:

        fallback_chunks = retrieve_chunks(
            original_query,
            k=6,
            expand=False
        )

        base_answer = generate_grounded(
            original_query,
            fallback_chunks
        )

        final_answer = (
            "Note: bridge analysis unavailable. "
            "Standard RAG response:\n\n"
            + base_answer
        )

        if state.get("degraded_mode", False):

            errors = state.get("errors", [])

            degraded_header = (
                "[DEGRADED MODE]\n"
                f"Encountered errors: {', '.join(errors)}\n\n"
            )

            final_answer = degraded_header + final_answer

        llm_calls_used += 2

        return {
            **state,
            "final_answer": final_answer,
            "llm_calls_used": llm_calls_used,
            "phase": "done",
        }

    # =================================
    # Collect grounded chunks
    # =================================

    collected_chunks = unique_chunks_from_evidence(
        evidence_records
    )

    # =================================
    # LLM Call #1:
    # grounded base answer
    # =================================

    base_answer = generate_grounded(
        original_query,
        collected_chunks
    )

    llm_calls_used += 1

    # =================================
    # Highest novelty bridge
    # =================================

    highest_novelty = None

    if bridge_candidates:
        highest_novelty = max(
            bridge_candidates,
            key=lambda c: c.novelty_score
        )

    # =================================
    # Build summaries
    # =================================

    evidence_summary = build_evidence_summary(
        evidence_records
    )

    bridge_summary = build_bridge_summary(
        bridge_candidates
    )

    system_notes = build_system_notes(state)

    # =================================
    # LLM Call #2:
    # structured synthesis
    # =================================

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        question=original_query,
        base_answer=base_answer,
        evidence_summary=evidence_summary,
        bridge_summary=bridge_summary,
        system_notes=system_notes,
    )

    structured_answer = invoke(
        [
            {
                "role": "user",
                "content": synthesis_prompt
            }
        ],
        max_new_tokens=1024,
        json_mode=False,
    )

    llm_calls_used += 1

    # =================================
    # Prepend degraded warning
    # =================================

    if state.get("degraded_mode", False):

        errors = state.get("errors", [])

        degraded_header = (
            "[DEGRADED MODE]\n"
            f"Encountered errors: {', '.join(errors)}\n\n"
        )

        structured_answer = (
            degraded_header + structured_answer
        )

    # =================================
    # Return updated state
    # =================================

    return {
        **state,
        "final_answer": structured_answer,
        "llm_calls_used": llm_calls_used,
        "phase": "done",
    }