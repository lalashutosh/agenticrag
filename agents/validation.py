# agents/validation.py

import json
from typing import List

from agents.state import (
    AgentState,
    BridgeCandidate,
    EvidenceRecord,
)

from core.rag_tools import (
    retrieve_chunks,
    RetrievedChunk,
)

from core.llm import invoke

# =========================
# Prompt Template
# =========================

VALIDATION_PROMPT = """You are evaluating evidence for a scientific hypothesis.

Hypothesis: Does "{subject_a}" connect to "{subject_b}" via "{bridge_concept}"?

Supporting evidence:
{supporting}

Opposing evidence:
{opposing}

Instructions:
- Output ONLY a JSON object, nothing else
- No markdown, no explanation, no preamble
- If supporting and opposing are comparable, use "speculative"
- Base verdict only on provided evidence, not prior knowledge

Required output format (copy exactly, fill in values):
{{"confidence": 0.0, "verdict": "speculative"}}

Valid verdict values: supported / speculative / contradicted
Confidence range: 0.0 to 1.0"""

# =========================
# Helpers
# =========================

def format_chunks(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "None"

    formatted = []

    for i, chunk in enumerate(chunks, start=1):
        formatted.append(
            f"[Chunk {i}] [Source: {chunk.source}]\n"
            f"{chunk.content}"
        )

    return "\n\n".join(formatted)


def safe_parse_validation_response(response_text: str):
    """
    Parse validation JSON safely.

    Returns:
        (confidence, verdict, parse_failed)
    """

    try:
        parsed = json.loads(response_text)

        confidence = float(parsed.get("confidence", 0.5))
        verdict = parsed.get("verdict", "speculative")

        if verdict not in {
            "supported",
            "speculative",
            "contradicted"
        }:
            verdict = "speculative"

        confidence = max(0.0, min(1.0, confidence))

        return confidence, verdict, False

    except Exception:
        return 0.5, "speculative", True


# =========================
# Main Node
# =========================

def validate_bridges(state: AgentState) -> AgentState:
    """
    Validation agent:
    - Adversarial retrieval
    - Supporting vs opposing evidence gathering
    - LLM grading

    IMPORTANT:
    retrieve_chunks(..., expand=False)
    makes ZERO LLM calls.
    Validation therefore performs:
        <= 3 LLM calls total
    (one grading call per candidate)
    """

    bridge_candidates = state.get("bridge_candidates", [])
    errors = list(state.get("errors", []))
    evidence_records: List[EvidenceRecord] = []

    llm_calls_used = state.get("llm_calls_used", 0)

    # ---------------------------------
    # Top 3 candidates only
    # ---------------------------------

    top_candidates = bridge_candidates[:3]

    for candidate in top_candidates:

        # =================================
        # Supporting retrieval
        # =================================

        supporting_query = (
            f"{candidate.triple_a.subject} "
            f"{candidate.bridge_concept} "
            f"{candidate.triple_b.subject}"
        )

        supporting_chunks = retrieve_chunks(
            supporting_query,
            k=4,
            expand=False,   # IMPORTANT: no query expansion here
        )

        # =================================
        # Opposing retrieval
        # =================================

        opposing_query = (
            f"limitations challenges "
            f"{candidate.bridge_concept} quantum"
        )

        opposing_chunks = retrieve_chunks(
            opposing_query,
            k=4,
            expand=False,   # IMPORTANT: no query expansion here
        )

        # =================================
        # Build validation prompt
        # =================================

        prompt = VALIDATION_PROMPT.format(
            subject_a=candidate.triple_a.subject,
            subject_b=candidate.triple_b.subject,
            bridge_concept=candidate.bridge_concept,
            supporting=format_chunks(supporting_chunks),
            opposing=format_chunks(opposing_chunks),
        )

        # =================================
        # Grading LLM call
        # =================================

        response = invoke(
            [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_new_tokens=256,
            json_mode=True,
        )

        llm_calls_used += 1

        # =================================
        # Parse response
        # =================================

        confidence, verdict, parse_failed = (
            safe_parse_validation_response(response)
        )

        if parse_failed:
            errors.append(
                f"validation: parse_failed for "
                f"{candidate.bridge_concept}"
            )

        # =================================
        # Build evidence record
        # =================================

        evidence = EvidenceRecord(
            candidate=candidate,
            supporting_chunks=supporting_chunks,
            opposing_chunks=opposing_chunks,
            confidence=confidence,
            verdict=verdict,
        )

        evidence_records.append(evidence)

    # ---------------------------------
    # Return updated state
    # ---------------------------------

    return {
        **state,
        "evidence": evidence_records,
        "llm_calls_used": llm_calls_used,
        "errors": errors,
        "phase": "synthesis",
    }