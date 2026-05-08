# agents/neighbor.py

import os
import os
from typing import Dict, List
from collections import defaultdict

from agents.state import (
    AgentState,
    Triple,
    NeighborResult
)

from core.rag_tools import vector_search

# =========================================================
# Tunable threshold (will be recalibrated after M3 ingest)
# =========================================================

NEIGHBOR_DISTANCE_THRESHOLD = 0.28


# =========================================================
# Utilities
# =========================================================

def _make_triple_key(triple: Triple) -> str:
    return f"{triple.subject}_{triple.object}"


def _build_search_string(triple: Triple) -> str:
    return f"{triple.subject} {triple.relation} {triple.object}"


def _to_cosine_distance(score: float) -> float:
    """
    vector_search returns cosine distance already in M3 version,
    but we keep this wrapper to isolate future changes.
    """
    return float(score)


# =========================================================
# Main Node
# =========================================================

def find_neighbors(state: AgentState) -> AgentState:

    triples = state.get("triples", [])
    errors = list(state.get("errors", []))

    # -----------------------------------------------------
    # No triples fallback
    # -----------------------------------------------------

    if not triples:
        errors.append("neighbor: skipped — no triples")

        return {
            **state,
            "neighbor_map": {},
            "errors": errors,
            "phase": "bridge"
        }

    # -----------------------------------------------------
    # Main neighbor map
    # -----------------------------------------------------

    neighbor_map: Dict[str, List[NeighborResult]] = {}

    for triple in triples:

        query = _build_search_string(triple)

        # IMPORTANT: dense-only semantic neighborhood
        raw_results = vector_search(query_text=query, k=6)

        triple_key = _make_triple_key(triple)

        filtered: List[NeighborResult] = []

        for chunk in raw_results:

            cosine_distance = _to_cosine_distance(chunk.score)

            # -------------------------------------------------
            # semantic proximity filter
            # -------------------------------------------------

            if cosine_distance > NEIGHBOR_DISTANCE_THRESHOLD:
                continue

            filtered.append(
                NeighborResult(
                    triple_key=triple_key,
                    chunk=chunk,
                    cosine_distance=cosine_distance
                )
            )

        if filtered:
            neighbor_map[triple_key] = filtered

    # -----------------------------------------------------
    # Corpus coverage signal
    # -----------------------------------------------------

    if not neighbor_map:
        errors.append("neighbor: no_neighbors_found")

    return {
        **state,
        "neighbor_map": neighbor_map,
        "errors": errors,
        "phase": "bridge"
    }


# =========================================================
# Calibration utility (IMPORTANT after M3 ingest)
# =========================================================

if __name__ == "__main__":

    import sys
    sys.path.insert(0, ".")

    from core.rag_tools import vector_search

    test_queries = [
        "quantum kernel feature map",
        "barren plateau gradient vanishing",
        "variational quantum circuit ansatz",
        "QAOA combinatorial optimization",
        "quantum advantage machine learning"
    ]

    print("Calibrating NEIGHBOR_DISTANCE_THRESHOLD (M3)\n")

    all_distances = []

    for q in test_queries:

        results = vector_search(q, k=10)
        distances = [r.score for r in results]

        all_distances.extend(distances)

        print(f"Query: {q}")
        print("  distances:", [f"{d:.3f}" for d in distances[:5]])

    all_distances.sort()

    p25 = all_distances[len(all_distances)//4]
    p50 = all_distances[len(all_distances)//2]

    print("\n--- Calibration Result ---")
    print(f"P25 threshold suggestion: {p25:.3f}")
    print(f"Median distance: {p50:.3f}")
    print(f"Recommended: NEIGHBOR_DISTANCE_THRESHOLD = {p25:.2f}")