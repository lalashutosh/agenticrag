# agents/bridge.py

import re
from itertools import combinations
from typing import Dict, List, Set, Optional

from agents.state import (
    AgentState,
    Triple,
    NeighborResult,
    BridgeCandidate,
)

# =========================
# Tunable constants
# =========================

NOVELTY_THRESHOLD = 0.5

# TODO: Replace extract_bridge_concept with a proper keyphrase extractor
# e.g. KeyBERT or spaCy noun chunk extraction

# =========================
# Triple normalization
# =========================

def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())

def triple_key(triple: Triple) -> str:
    """Must match exactly how neighbor.py builds its keys."""
    return f"{triple.subject}_{triple.object}"

def deduplicate_triples(triples: List[Triple]) -> List[Triple]:
    """
    Remove triples where:
    1. (subject, object) pair is a duplicate (keep first occurrence)
    2. subject == object after normalization (self-referential, useless for bridging)
    3. subject or object is empty/whitespace
    """
    seen_keys: Set[tuple] = set()
    result: List[Triple] = []

    for t in triples:
        subj = normalize(t.subject)
        obj  = normalize(t.object)

        # Filter empty
        if not subj or not obj:
            continue

        # Filter self-referential
        if subj == obj:
            continue

        # Filter duplicate (subject, object) pairs
        pair = (subj, obj)
        if pair in seen_keys:
            continue

        seen_keys.add(pair)
        result.append(t)

    return result

# =========================
# Bridge concept extraction
# =========================

def extract_bridge_concept(chunk_content: str) -> str:
    """
    Extract the most informative noun phrase from a chunk.
    No LLM call — pure heuristic.

    Strategy:
    1. Take first two sentences
    2. Strip common academic preamble
    3. Walk words until hitting a verb, take what's before it
    4. Fallback to first 60 chars if result is too short
    """

    # First two sentences
    sentences = re.split(r'[.!?]', chunk_content)
    text = ' '.join(sentences[:2]).strip()

    # Strip academic preamble
    text = re.sub(
        r'^(we show|we prove|in this paper|it is known|there is|the|a|an)\s+',
        '', text, flags=re.IGNORECASE
    )

    # Walk until verb
    stop_verbs = {
        'is', 'are', 'was', 'were', 'can', 'may', 'might',
        'have', 'has', 'had', 'show', 'shows', 'showed',
        'prove', 'proves', 'proved', 'demonstrate', 'demonstrates'
    }

    words = text.split()[:14]
    concept_words: List[str] = []

    for w in words:
        clean = w.lower().strip('.,;:()')
        if clean in stop_verbs and len(concept_words) >= 2:
            break
        concept_words.append(w)

    concept = ' '.join(concept_words[:8]).strip('.,;:()')

    # Fallback
    if len(concept.split()) < 2:
        return chunk_content[:60].strip()

    return concept

# =========================
# Neighbor map inspection
# =========================

def get_neighbor_sources(neighbor_list: List[NeighborResult]) -> Set[str]:
    return {n.chunk.source for n in neighbor_list}

def build_source_lookup(neighbor_list: List[NeighborResult]) -> Dict[str, NeighborResult]:
    """
    source → best (lowest distance) NeighborResult for that source.
    If a source appears multiple times, keep the closest match.
    """
    lookup: Dict[str, NeighborResult] = {}
    for n in neighbor_list:
        existing = lookup.get(n.chunk.source)
        if existing is None or n.cosine_distance < existing.cosine_distance:
            lookup[n.chunk.source] = n
    return lookup

def count_co_occurrences(
    neighbor_list_a: List[NeighborResult],
    neighbor_list_b: List[NeighborResult],
    subject_a: str,
    subject_b: str,
) -> int:
    """
    Count chunks (across both neighbor lists) that contain
    BOTH subject_a and subject_b as substrings.
    This measures how often these concepts appear together —
    high co-occurrence = already explored, low = novel bridge.
    """
    subj_a = normalize(subject_a)
    subj_b = normalize(subject_b)

    count = 0
    # Check all chunks from both lists (deduplicated by content)
    seen_content: Set[str] = set()

    for n in neighbor_list_a + neighbor_list_b:
        content = normalize(n.chunk.content)

        # Skip duplicate chunks
        content_key = content[:100]  # first 100 chars as proxy key
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        if subj_a in content and subj_b in content:
            count += 1

    return count

# =========================
# Main node
# =========================

def find_bridges(state: AgentState) -> dict:
    raw_triples:  List[Triple]                       = state.get("triples", [])
    neighbor_map: Dict[str, List[NeighborResult]]    = state.get("neighbor_map", {})
    errors:       List[str]                          = list(state.get("errors", []))

    # --------------------------------------------------
    # Guard: nothing to work with
    # --------------------------------------------------

    if not neighbor_map:
        errors.append("bridge: skipped — empty neighbor_map")
        return {
            **state,
            "bridge_candidates": [],
            "errors": errors,
            "phase": "validation",
        }

    # --------------------------------------------------
    # Step 1: Deduplicate triples
    # --------------------------------------------------

    triples = deduplicate_triples(raw_triples)

    if len(triples) < 2:
        errors.append(
            f"bridge: insufficient distinct triples after dedup "
            f"(had {len(raw_triples)}, kept {len(triples)})"
        )
        return {
            **state,
            "bridge_candidates": [],
            "errors": errors,
            "phase": "validation",
        }

    # --------------------------------------------------
    # Step 2: Filter to triples that actually have
    #         neighbors in the map (inner join)
    # --------------------------------------------------

    triples_with_neighbors = [
        t for t in triples
        if triple_key(t) in neighbor_map
           and len(neighbor_map[triple_key(t)]) > 0
    ]

    if len(triples_with_neighbors) < 2:
        errors.append(
            f"bridge: fewer than 2 triples have neighbors "
            f"(triples={len(triples)}, with_neighbors={len(triples_with_neighbors)}). "
            f"Available neighbor keys: {list(neighbor_map.keys())[:5]}"
        )
        return {
            **state,
            "bridge_candidates": [],
            "errors": errors,
            "phase": "validation",
        }

    # --------------------------------------------------
    # Step 3: Pairwise bridge detection
    # --------------------------------------------------

    bridge_candidates: List[BridgeCandidate] = []

    for triple_a, triple_b in combinations(triples_with_neighbors, 2):

        key_a = triple_key(triple_a)
        key_b = triple_key(triple_b)

        neighbor_list_a = neighbor_map[key_a]
        neighbor_list_b = neighbor_map[key_b]

        # Build source → best neighbor lookups for both sides
        lookup_a = build_source_lookup(neighbor_list_a)
        lookup_b = build_source_lookup(neighbor_list_b)

        # Find papers that appear as neighbors of BOTH triples
        shared_sources = set(lookup_a.keys()).intersection(set(lookup_b.keys()))

        if not shared_sources:
            continue

        # --------------------------------------------------
        # Step 4: Co-occurrence check
        # Shared source alone doesn't mean the concepts
        # appear together — verify at chunk content level
        # --------------------------------------------------

        co_occurrence_count = count_co_occurrences(
            neighbor_list_a,
            neighbor_list_b,
            triple_a.subject,
            triple_b.subject,
        )

        total_chunks = len(neighbor_list_a) + len(neighbor_list_b)
        novelty_score = (
            1.0 - (co_occurrence_count / total_chunks)
            if total_chunks > 0
            else 0.0
        )

        if novelty_score <= NOVELTY_THRESHOLD:
            continue

        # --------------------------------------------------
        # Step 5: One BridgeCandidate per shared source
        # Use the chunk from whichever side had better
        # (lower) cosine distance for that source
        # --------------------------------------------------

        for source in shared_sources:
            neighbor_a = lookup_a[source]
            neighbor_b = lookup_b[source]

            # Pick the chunk from the side with stronger signal
            best_neighbor = (
                neighbor_a
                if neighbor_a.cosine_distance <= neighbor_b.cosine_distance
                else neighbor_b
            )

            bridge_concept = extract_bridge_concept(best_neighbor.chunk.content)

            candidate = BridgeCandidate(
                triple_a=triple_a,
                triple_b=triple_b,
                bridge_concept=bridge_concept,
                co_occurrence_count=co_occurrence_count,
                novelty_score=float(novelty_score),
            )

            bridge_candidates.append(candidate)

    # --------------------------------------------------
    # Step 6: Deduplicate candidates
    # Same (triple_a, triple_b) pair can appear multiple
    # times if they share multiple sources — keep best
    # --------------------------------------------------

    deduped_candidates: Dict[tuple, BridgeCandidate] = {}

    for c in bridge_candidates:
        pair_key = (
            normalize(c.triple_a.subject),
            normalize(c.triple_b.subject),
        )
        existing = deduped_candidates.get(pair_key)
        if existing is None or c.novelty_score > existing.novelty_score:
            deduped_candidates[pair_key] = c

    final_candidates = sorted(
        deduped_candidates.values(),
        key=lambda c: c.novelty_score,
        reverse=True,
    )

    # --------------------------------------------------
    # Error logging
    # --------------------------------------------------

    if not final_candidates:
        errors.append(
            f"bridge: no_candidates (checked {len(triples_with_neighbors)} triples, "
            f"{sum(len(v) for v in neighbor_map.values())} total neighbors)"
        )

    # --------------------------------------------------
    # Debug summary (remove in production)
    # --------------------------------------------------

    print(f"[bridge] raw_triples={len(raw_triples)} "
          f"→ deduped={len(triples)} "
          f"→ with_neighbors={len(triples_with_neighbors)} "
          f"→ candidates={len(final_candidates)}")

    for c in final_candidates[:3]:
        print(f"  [{c.novelty_score:.2f}] "
              f"{c.triple_a.subject!r} ↔ {c.triple_b.subject!r} "
              f"via {c.bridge_concept!r}")

    return {
        **state,
        "bridge_candidates": final_candidates,
        "errors": errors,
        "phase": "validation",
    }