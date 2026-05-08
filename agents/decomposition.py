# agents/decomposition.py

import json
from typing import List, Tuple

from agents.state import AgentState, Triple
from core.rag_tools import retrieve_chunks
from core.llm import invoke


SYSTEM_PROMPT = """
You extract knowledge graph triples from scientific text.

Output only a JSON array of objects with keys:
- subject
- relation
- object
- source

source must be:
- "query" if the triple came directly from the question
- "retrieved" if it came from retrieved context
"""


def _build_prompt(query: str, chunk_text: str) -> str:
    """
    Construct extraction prompt with QML few-shot examples.
    """

    return f"""
Extract scientific knowledge graph triples from the material below.

Few-shot examples:

[
  {{
    "subject": "quantum kernel",
    "relation": "maps to",
    "object": "Hilbert space",
    "source": "retrieved"
  }},
  {{
    "subject": "VQE",
    "relation": "minimizes",
    "object": "expectation value",
    "source": "retrieved"
  }}
]

Instructions:
- Maximum 12 triples
- Prefer precise scientific relations
- Avoid generic relations like "uses", "has", "related to"
- Output ONLY valid JSON
- source="query" only if directly stated in the question
- source="retrieved" if extracted from retrieved context

Question:
{query}

Retrieved Context:
{chunk_text}
"""


def _deduplicate(
    triples: List[Triple]
) -> List[Triple]:
    """
    Deduplicate by (subject, object).
    Keeps first occurrence.
    """

    seen: set[Tuple[str, str]] = set()

    deduped: List[Triple] = []

    for t in triples:

        key = (
            t.subject.strip().lower(),
            t.object.strip().lower()
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(t)

    return deduped


def decompose(state: AgentState) -> AgentState:
    """
    Decomposition node.

    Flow:
    1. Retrieve chunks
    2. Build extraction prompt
    3. Single JSON-mode LLM extraction call
    4. Parse triples
    5. Deduplicate
    """

    query = state.get("original_query", "")

    errors = list(state.get("errors", []))

    llm_calls = state.get("llm_calls_used", 0)

    retrieved_log = list(
        state.get("retrieved_chunks_log", [])
    )

    # =====================================================
    # 1. Retrieve
    # =====================================================

    chunks = retrieve_chunks(
        query=query,
        k=4,
        expand=True
    )

    # Track sources used
    retrieved_log.extend([
        c.source for c in chunks
    ])

    # =====================================================
    # 2. Build context
    # =====================================================

    chunk_text = "\n\n".join(
        c.content for c in chunks
    )

    user_prompt = _build_prompt(
        query=query,
        chunk_text=chunk_text
    )

    # =====================================================
    # 3. SINGLE LLM CALL
    # =====================================================

    response = invoke(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_new_tokens=512,
        json_mode=True
    )

    llm_calls += 1

    # =====================================================
    # 4. Parse
    # =====================================================

    parsed_triples: List[Triple] = []

    try:

        data = json.loads(response)

        if not isinstance(data, list):
            raise ValueError(
                "Expected JSON list"
            )

        for item in data:

            if not isinstance(item, dict):
                continue

            triple = Triple(
                subject=str(
                    item.get("subject", "")
                ).strip(),

                relation=str(
                    item.get("relation", "")
                ).strip(),

                object=str(
                    item.get("object", "")
                ).strip(),

                source=(
                    "query"
                    if item.get("source") == "query"
                    else "retrieved"
                )
            )

            # Skip malformed entries
            if (
                not triple.subject or
                not triple.relation or
                not triple.object
            ):
                continue

            parsed_triples.append(triple)

        parsed_triples = _deduplicate(
            parsed_triples
        )

    except Exception:

        errors.append(
            "decomposition: json_parse_failed"
        )

        parsed_triples = []

    # =====================================================
    # 5. Return updated state
    # =====================================================

    return {
        **state,

        "triples": parsed_triples,

        "llm_calls_used": llm_calls,

        "retrieved_chunks_log": retrieved_log,

        "errors": errors,

        "phase": "neighbor"
    }