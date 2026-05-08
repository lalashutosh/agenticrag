# testing/test_quality.py

import sys, json
sys.path.insert(0, ".")

from agents.decomposition import decompose
from agents.neighbor import find_neighbors
from agents.bridge import find_bridges
from agents.state import AgentState

# ── Probe 1: Are triples actually specific? ───────────────────────────────────

def probe_triple_specificity():
    """
    Generic triples like ("quantum", "uses", "circuit") are useless for bridge finding.
    Good triples have specific technical subjects and relations.
    """
    print("\nPROBE: Triple Specificity")
    query = "How do quantum kernels relate to classical support vector machines?"

    state = AgentState(
        original_query=query, triples=[], neighbor_map={},
        bridge_candidates=[], evidence=[], final_answer="",
        phase="init", degraded_mode=False, errors=[],
        llm_calls_used=0, retrieved_chunks_log=[]
    )
    state.update(decompose(state))

    print(f"  Query: {query}")
    print(f"  Triples extracted ({len(state['triples'])}):")
    for t in state["triples"]:
        length_ok = len(t.subject) > 3 and len(t.object) > 3
        relation_ok = t.relation not in {"uses", "has", "is", "does"}
        flag = "✓" if (length_ok and relation_ok) else "⚠ generic"
        print(f"    {flag}  ({t.subject}) --[{t.relation}]--> ({t.object})")

# ── Probe 2: Are neighbors actually relevant? ─────────────────────────────────

def probe_neighbor_relevance():
    """
    Neighbors should be thematically related, not just lexically similar.
    Print top neighbor for each triple so you can read and judge.
    """
    print("\nPROBE: Neighbor Relevance")
    from agents.state import Triple
    from agents.neighbor import find_neighbors

    triples = [
        Triple(subject="quantum kernel", relation="estimates", object="inner product in Hilbert space", source="query"),
        Triple(subject="barren plateau", relation="vanishes", object="gradient in variational circuit", source="query"),
    ]

    state = AgentState(
        original_query="", triples=triples, neighbor_map={},
        bridge_candidates=[], evidence=[], final_answer="",
        phase="neighbor", degraded_mode=False, errors=[],
        llm_calls_used=0, retrieved_chunks_log=[]
    )
    state.update(find_neighbors(state))

    for key, neighbors in state["neighbor_map"].items():
        print(f"\n  Triple key: {key}")
        print(f"  Top 3 neighbors:")
        for n in neighbors[:3]:
            print(f"    dist={n.cosine_distance:.3f}  src={n.chunk.source}")
            print(f"    '{n.chunk.content[:120]}...'")

# ── Probe 3: Is the bridge non-obvious? ──────────────────────────────────────

def probe_bridge_novelty():
    """
    The key question: are bridge candidates things that AREN'T already 
    in the same papers? High novelty_score + low co_occurrence_count = good.
    """
    print("\nPROBE: Bridge Novelty")
    query = "What quantum algorithms could exploit kernel-induced feature spaces?"

    state = AgentState(
        original_query=query, triples=[], neighbor_map={},
        bridge_candidates=[], evidence=[], final_answer="",
        phase="init", degraded_mode=False, errors=[],
        llm_calls_used=0, retrieved_chunks_log=[]
    )

    from agents.decomposition import decompose
    from agents.neighbor import find_neighbors
    from agents.bridge import find_bridges

    state.update(decompose(state))
    state.update(find_neighbors(state))
    state.update(find_bridges(state))

    print(f"  Candidates found: {len(state['bridge_candidates'])}")
    for c in state["bridge_candidates"][:3]:
        print(f"\n  Bridge: {c.triple_a.subject} ←→ {c.triple_b.subject}")
        print(f"  Via:    {c.bridge_concept}")
        print(f"  Novelty score:       {c.novelty_score:.3f}  (higher = less explored)")
        print(f"  Co-occurrence count: {c.co_occurrence_count}  (lower = more novel)")

        # Flag if the bridge is too obvious
        if c.co_occurrence_count > 5:
            print("  ⚠  High co-occurrence — this connection may already be well-studied")
        elif c.novelty_score > 0.8:
            print("  ✓  Strong novelty signal")

# ── Probe 4: End-to-end answer structure ─────────────────────────────────────

def probe_answer_structure():
    """
    Run the full pipeline and check the answer has the right epistemic structure.
    Print word counts per section — a section with < 20 words is probably empty.
    """
    print("\nPROBE: Answer Structure")
    import re
    from agents.validation import validate_bridges
    from agents.synthesis import synthesize

    query = "Can barren plateaus be mitigated using kernel-based initialization strategies?"

    state = AgentState(
        original_query=query, triples=[], neighbor_map={},
        bridge_candidates=[], evidence=[], final_answer="",
        phase="init", degraded_mode=False, errors=[],
        llm_calls_used=0, retrieved_chunks_log=[]
    )

    for fn in [decompose, find_neighbors, find_bridges, validate_bridges, synthesize]:
        state.update(fn(state))

    answer = state["final_answer"]
    sections = ["ESTABLISHED", "PROBABLE BRIDGES", "OPEN QUESTION", "LIMITATIONS"]

    print(f"\n  Total words: {len(answer.split())}")
    print(f"  LLM calls:  {state['llm_calls_used']}")
    print(f"  Errors:     {state['errors']}")
    print()

    for s in sections:
        if s in answer:
            # Extract section content until next section or end
            pattern = rf"{s}.*?(?={'|'.join(sections[sections.index(s)+1:])}|$)"
            match = re.search(pattern, answer, re.DOTALL)
            words = len(match.group().split()) if match else 0
            flag = "✓" if words > 20 else "⚠ thin"
            print(f"  {flag}  {s}: ~{words} words")
        else:
            print(f"  ✗  {s}: MISSING")

    print(f"\n--- FULL ANSWER ---\n{answer}\n---")

if __name__ == "__main__":
    print("=" * 60)
    print("QUALITY PROBES — read the output, these are not pass/fail")
    print("=" * 60)

    probe_triple_specificity()
    probe_neighbor_relevance()
    probe_bridge_novelty()
    probe_answer_structure()